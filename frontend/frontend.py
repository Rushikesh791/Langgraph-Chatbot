import sys
import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


# --------------------------------------------------
# Path setup (important for Streamlit)
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.backend import chatbot

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
    st.session_state.thread_ids = []

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}

if "current_thread_id" not in st.session_state:
    first_thread = generate_thread_id()
    st.session_state.current_thread_id = first_thread
    st.session_state.thread_ids.append(first_thread)
    st.session_state.conversations[first_thread] = []
    st.session_state.thread_titles[first_thread] = "New Chat"

# --------------------------------------------------
# Sidebar UI
# --------------------------------------------------
st.sidebar.title("🧠 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    new_thread = generate_thread_id()
    st.session_state.thread_ids.append(new_thread)
    st.session_state.conversations[new_thread] = []
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
message_history = st.session_state.conversations[thread_id]

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
    lc_messages = []
    for msg in message_history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    CONFIG = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    response = chatbot.invoke(
        {"messages": lc_messages},
        config=CONFIG
    )

    ai_text = response["messages"][-1].content

    # Store assistant message
    message_history.append(
        {"role": "assistant", "content": ai_text}
    )

    with st.chat_message("assistant"):
        st.markdown(ai_text)

    # --------------------------------------------------
    # Auto-generate title (only once)
    # --------------------------------------------------
    if st.session_state.thread_titles.get(thread_id) == "New Chat":
        try:
            title = generate_title(lc_messages)
            st.session_state.thread_titles[thread_id] = title
        except Exception:
            pass  # fail silently if title generation fails
