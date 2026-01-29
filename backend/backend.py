# from langgraph.graph import StateGraph , START, END
# from typing import TypedDict, Literal, Annotated
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage, BaseMessage
# import operator
# from langchain_huggingface import HuggingFacePipeline
# # from langchain.chat_models import ChatHuggingFace


# # llm = ChatOpenAI()

# from transformers import pipeline
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv

# load_dotenv()

# from langgraph.graph.message import add_messages

# class ChatState(TypedDict):
#     messages : Annotated[list[BaseMessage], add_messages]

# hf_pipeline = pipeline(
#     task="text2text-generation",
#     model="deepseek-ai/deepseek-coder-1.3b-instruct",                     #mistralai/Mistral-7B-Instruct-v0.2   meta-llama/Llama-3.1-8B-Instruct
#     max_new_tokens=256,
#     temperature=0.0,         
#     do_sample=False,          
#     repetition_penalty=1.1,
#     return_full_text=False,
#     device=-1)

# llm = HuggingFacePipeline(
#     pipeline=hf_pipeline)

# # def chat_node(state : ChatState):
# #     messages = state['messages']

# #     response = llm.invoke(messages)

# #     return { 'messages':[response]}
# def chat_node(state: ChatState):
#     messages = state["messages"]

#     prompt = ""
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             prompt += f"User: {msg.content}\n"
#         elif isinstance(msg, AIMessage):
#             prompt += f"Assistant: {msg.content}\n"

#     prompt += "Assistant:"

#     response = llm.invoke(prompt)

#     return {"messages": [AIMessage(content=response)]}


# graph = StateGraph(ChatState)

# graph.add_node("chat_node",chat_node)

# graph.add_edge(START, "chat_node")
# graph.add_edge("chat_node",END)

# chatbot= graph.compile()


# initial_state = {
#     'messages': [HumanMessage(content='''Explain machine learning in simple terms.
# Use bullet points.
# Do not exceed 150 words.''')]
# }

# final_state = chatbot.invoke(initial_state)["messages"][-1].content

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

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()


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
    # inputs = tokenizer(prompt, return_tensors="pt")

    # streamer = TextIteratorStreamer(
    #     tokenizer,
    #     skip_prompt=True,
    #     skip_special_tokens=True
    # )

    # generation_kwargs = dict(
    #     **inputs,
    #     streamer=streamer,
    #     max_new_tokens=256,
    #     temperature=0.0,
    #     do_sample=False,
    #     repetition_penalty=1.1,
    # )

    # # Run generation in background thread
    # thread = threading.Thread(
    #     target=model.generate,
    #     kwargs=generation_kwargs
    # )
    # thread.start()

    # # 🔥 Yield tokens as they arrive
    # for token in streamer:
    #     yield {
    #         "messages": [AIMessage(content=token)]
    #     }

# ------------------ GRAPH ------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()








