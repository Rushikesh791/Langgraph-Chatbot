# --------------------------------------------------
# Path setup (important for Streamlit)
# --------------------------------------------------
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
from backend.backend import chatbot, retrieve_all_threads
from backend.metadata import save_thread_title, load_all_thread_titles



# --------------------------------------------------
# Helpers
# --------------------------------------------------
def generate_thread_id():
    return f"thread-{uuid.uuid4()}"

def generate_title(messages):
    """
    Generate a short title (max 5 words) for a conversation
    """
    prompt = [
        HumanMessage(
            content=(
                "Generate a short, clear title (maximum 5 words) "
                "for this conversation. No quotes, no punctuation."
            )
        )
    ]

    prompt.extend(messages[:4])  # first few messages only

    response = chatbot.invoke(
        {"messages": prompt},
        config={"configurable": {"thread_id": "title-generator"}}
    )

    return response["messages"][-1].content.strip()




# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "thread_ids" not in st.session_state:
    db_threads = retrieve_all_threads()

    if db_threads:
        st.session_state.thread_ids = db_threads
    else:
        first_thread = generate_thread_id()
        st.session_state.thread_ids = [first_thread]

    st.session_state.current_thread_id = st.session_state.thread_ids[0]


# if "conversations" not in st.session_state:
#     st.session_state.conversations = {}

# if "thread_titles" not in st.session_state:
#     st.session_state.thread_titles = {}
if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = load_all_thread_titles()


if "current_thread_id" not in st.session_state:
    first_thread = generate_thread_id()
    st.session_state.current_thread_id = first_thread
    st.session_state.thread_ids.append(first_thread)
    st.session_state.thread_titles[first_thread] = "New Chat"

# --------------------------------------------------
# Sidebar UI
# --------------------------------------------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("+ New Chat"):
    new_thread = generate_thread_id()
    st.session_state.thread_ids.append(new_thread)
    st.session_state.thread_titles[new_thread] = "New Chat"
    st.session_state.current_thread_id = new_thread

st.sidebar.divider()
st.sidebar.header("My Conversations")

for tid in st.session_state.thread_ids:
    title = st.session_state.thread_titles.get(tid, tid)
    if st.sidebar.button(title, key=tid):
        st.session_state.current_thread_id = tid

# --------------------------------------------------
# Active Conversation
# --------------------------------------------------
thread_id = st.session_state.current_thread_id
# message_history = st.session_state.conversations[thread_id]
state = chatbot.get_state(
    config={"configurable": {"thread_id": thread_id}})

message_history = []

if state and "messages" in state.values:
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            message_history.append(
                {"role": "user", "content": msg.content}
            )
        elif isinstance(msg, AIMessage):
            message_history.append(
                {"role": "assistant", "content": msg.content}
            )


# Display existing messages
for msg in message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert history to LangChain messages
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )


    # --------------------------------------------------
    # Auto-generate title (only once)
    # --------------------------------------------------
    
    

    if st.session_state.thread_titles.get(thread_id) == "New Chat":
        try:
            state = chatbot.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )

            messages = state.values.get("messages", [])

            if len(messages) >= 2:  # user + AI
                title = generate_title(messages)
                save_thread_title(thread_id, title)
                st.session_state.thread_titles[thread_id] = title

        except Exception as e:
            print("Title generation failed:", e)
