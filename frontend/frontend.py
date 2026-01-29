# import streamlit as st
# from backend import chatbot
# from langchain_core.messages import HumanMessage
# from langchain_core.messages import HumanMessage, AIMessage


# # st.session_state -> dict -> 
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []

# # loading the conversation history
# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.text(message['content'])


# #{'role': 'user', 'content': 'Hi'}
# #{'role': 'assistant', 'content': 'Hi=ello'}

# user_input = st.chat_input('Type here')

# if user_input:

#     # first add the message to message_history
#     messages = []

#     for msg in st.session_state['message_history']:
#         if msg['role'] == 'user':
#             messages.append(HumanMessage(content=msg['content']))
#         else:
#             messages.append(AIMessage(content=msg['content']))
#     # st.session_state['message_history'].append({'role': 'user', 'content': user_input})
#     # with st.chat_message('user'):
#     #     st.text(user_input)

#     # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
#     response = chatbot.invoke(
#             {'messages': messages + [HumanMessage(content=user_input)]},
#                 config=CONFIG)

#     ai_message = response['messages'][-1].content
#     # first add the message to message_history
#     st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
#     with st.chat_message('assistant'):
#         st.text(ai_message)



import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)





import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from backend.backend import chatbot
from backend import chatbot

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# display history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # show user message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # convert history → LangChain messages
    messages = []
    for msg in st.session_state.message_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    response = chatbot.invoke(
        {"messages": messages},
        config=CONFIG
    )

    ai_text = response["messages"][-1].content
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_text})
    
    with st.chat_message("assistant"):
        st.markdown(ai_text)

    # placeholder.markdown(token-by-token)

    # with st.chat_message("assistant"):
    #     placeholder = st.empty()
    #     streamed_text = ""

    #     for event in chatbot.stream(
    #         {"messages": messages},
    #         config=CONFIG):
    #         if "messages" in event:
    #             chunk = event["messages"][-1].content
    #             streamed_text += chunk
    #             placeholder.markdown(streamed_text)

    # # save final message
    # st.session_state.message_history.append(
    #     {"role": "assistant", "content": streamed_text})





