from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# Create thread names table if it doesn't exist
def init_thread_names_table():
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

# Initialize the table
init_thread_names_table()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def save_thread_name(thread_id, name):
    """Save or update a thread name in the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO thread_names (thread_id, name)
            VALUES (?, ?)
        ''', (str(thread_id), name))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error saving thread name: {e}")
        conn.rollback()
        return False

def update_thread_name(thread_id, new_name):
    """Update an existing thread name in the database"""
    return save_thread_name(thread_id, new_name)  # Same functionality as save

def get_thread_name(thread_id):
    """Get a thread name from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name FROM thread_names WHERE thread_id = ?
        ''', (str(thread_id),))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Error getting thread name: {e}")
        return None

def get_all_thread_names():
    """Get all thread names from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT thread_id, name FROM thread_names
        ''')
        results = cursor.fetchall()
        return {thread_id: name for thread_id, name in results}
    except sqlite3.Error as e:
        print(f"Error getting all thread names: {e}")
        return {}

def delete_thread_from_db(thread_id):
    """Delete a thread and all its checkpoints from the database"""
    try:
        cursor = conn.cursor()
        # Delete all checkpoints for the specific thread_id
        cursor.execute("""
            DELETE FROM checkpoints 
            WHERE thread_id = ?
        """, (str(thread_id),))
        
        # Also delete from writes table if it exists
        cursor.execute("""
            DELETE FROM writes 
            WHERE thread_id = ?
        """, (str(thread_id),))
        
        # Delete the thread name
        cursor.execute("""
            DELETE FROM thread_names 
            WHERE thread_id = ?
        """, (str(thread_id),))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error deleting thread from database: {e}")
        conn.rollback()
        return False