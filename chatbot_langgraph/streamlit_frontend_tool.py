import streamlit as st
from langgraph_tool_backend import (
    chatbot, retrieve_all_threads, delete_thread_from_db, 
    save_thread_name, get_thread_name, get_all_thread_names, update_thread_name
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# **************************************** Utility Functions *************************
def generate_thread_id():
    return uuid.uuid4()

def generate_chat_name(message):
    """Generate a chat name from the first message (first 30 chars)"""
    return message[:30] + "..." if len(message) > 30 else message

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

def delete_thread(thread_id):
    # Delete from database (includes checkpoints and thread name)
    success = delete_thread_from_db(thread_id)
    
    if success:
        # Remove from chat threads list
        if thread_id in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].remove(thread_id)
        
        # Refresh thread names from database
        st.session_state["chat_names"] = get_all_thread_names()
        
        # If we're deleting the current thread, start a new one
        if st.session_state["thread_id"] == thread_id:
            reset_chat()
        
        st.success("Chat deleted successfully!")
    else:
        st.error("Failed to delete chat from database")

# **************************************** Session Initialization *******************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Load thread names from database instead of empty dict
if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = get_all_thread_names()

# Track which thread is being edited
if "editing_thread" not in st.session_state:
    st.session_state["editing_thread"] = None

add_thread(st.session_state["thread_id"])

# **************************************** Sidebar UI ****************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    # Get thread name from database or use "Untitled"
    chat_display_name = st.session_state["chat_names"].get(thread_id, "Untitled")
    
    # Check if this thread is being edited
    is_editing = st.session_state["editing_thread"] == thread_id
    
    if is_editing:
        # Show text input for editing
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        
        with col1:
            new_name = st.text_input(
                "Edit name", 
                value=chat_display_name, 
                key=f"edit_input_{thread_id}",
                label_visibility="collapsed"
            )
        
        with col2:
            # Save button
            if st.button("✓", key=f"save_{thread_id}", help="Save name"):
                if new_name.strip():  # Only save if not empty
                    success = update_thread_name(thread_id, new_name.strip())
                    if success:
                        st.session_state["chat_names"][thread_id] = new_name.strip()
                        st.session_state["editing_thread"] = None
                        st.success("Name updated!")
                        st.rerun()
                    else:
                        st.error("Failed to update name")
        
        with col3:
            # Cancel button
            if st.button("✗", key=f"cancel_{thread_id}", help="Cancel"):
                st.session_state["editing_thread"] = None
                st.rerun()
    
    else:
        # Normal display mode
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        
        with col1:
            if st.button(chat_display_name, key=f"chat_{thread_id}"):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)

                temp_messages = []
                for msg in messages:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    temp_messages.append({"role": role, "content": msg.content})
                st.session_state["message_history"] = temp_messages
        
        with col2:
            # Edit button
            if st.button("✏️", key=f"edit_{thread_id}", help="Rename chat"):
                st.session_state["editing_thread"] = thread_id
                st.rerun()
        
        with col3:
            # Delete button
            if st.button("🗑️", key=f"delete_{thread_id}", help="Delete chat"):
                delete_thread(thread_id)
                st.rerun()

# **************************************** Main UI ****************************

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    # Generate chat name from first message and save to database
    if st.session_state["thread_id"] not in st.session_state["chat_names"]:
        chat_name = generate_chat_name(user_input)
        success = save_thread_name(st.session_state["thread_id"], chat_name)
        if success:
            # Update session state with the new name
            st.session_state["chat_names"][st.session_state["thread_id"]] = chat_name
    
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block with tool status
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    
    # Rerun to update the sidebar with the new chat name
    st.rerun()