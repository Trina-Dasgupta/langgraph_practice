import streamlit as st
from langgraph_threading import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def generate_chat_name(message):
    """Generate a chat name from the first message (first 30 chars)"""
    return message[:30] + "..." if len(message) > 30 else message

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def delete_thread(thread_id):
    # Remove from chat threads list
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    
    # Remove from chat names
    if thread_id in st.session_state['chat_names']:
        del st.session_state['chat_names'][thread_id]
    
    # If we're deleting the current thread, start a new one
    if st.session_state['thread_id'] == thread_id:
        reset_chat()


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'chat_names' not in st.session_state:
    st.session_state['chat_names'] = {}

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # Use chat name if available, otherwise use "Untitled" instead of thread_id
    chat_display_name = st.session_state['chat_names'].get(thread_id, "Untitled")
    
    # Create columns for chat name and delete button
    col1, col2 = st.sidebar.columns([4, 1])
    
    with col1:
        if st.button(chat_display_name, key=f"chat_{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role='user'
                else:
                    role='assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
    
    with col2:
        if st.button("🗑️", key=f"delete_{thread_id}", help="Delete chat"):
            delete_thread(thread_id)
            st.rerun()


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # Generate chat name from first message if not already set
    if st.session_state['thread_id'] not in st.session_state['chat_names']:
        st.session_state['chat_names'][st.session_state['thread_id']] = generate_chat_name(user_input)
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    # Rerun to update the sidebar with the new chat name
    st.rerun()