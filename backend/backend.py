from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


# ------------------ STATE ------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------ HF PIPELINE ------------------
hf_pipeline = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,
    temperature=0.0,        # deterministic
    do_sample=False,
    repetition_penalty=1.1,
    return_full_text=False,
    device=-1
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# ------------------ NODE ------------------
def chat_node(state: ChatState):
    messages = state["messages"]

    
    # prompt = "<|system|>\nYou are a helpful AI assistant.\n</s>\n"
    prompt = """<|system|>
                You are a helpful AI assistant.
                Always reply in English.
                Do not repeat system or user messages.
                </s>
            """


    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt += f"<|user|>\n{msg.content}\n</s>\n"
        elif isinstance(msg, AIMessage):
            prompt += f"<|assistant|>\n{msg.content}\n</s>\n"

    prompt += "<|assistant|>\n"

    response = llm.invoke(prompt)

    clean_response = (
    response.replace("</s>", "").replace("<|assistant|>", "").strip()
            )

    return {"messages": [AIMessage(content=clean_response)]}
   

# ------------------ GRAPH ------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()








