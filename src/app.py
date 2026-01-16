import config
import streamlit as st
import uuid
from datetime import datetime
from agents import query_agent_stream, list_all_threads, get_thread_first_message, delete_thread, get_thread_messages
from config import DUCKDB_PATH, CHROMA_PATH, OPENAI_API_KEY, LLM_MODEL

# STARTUP CHECK


# Check if data setup is complete
if not DUCKDB_PATH.exists() or not CHROMA_PATH.exists():
    st.error("Database setup incomplete!")
    st.info("""
    Please run the data setup first:
    
    ```bash
    uv run python src/data_setup.py
    ```
    
    This will:
    1. Load and clean the IMDB CSV data
    2. Create the DuckDB database
    3. Generate embeddings and create ChromaDB vector store
    """)
    st.stop()

if not OPENAI_API_KEY:
    st.error("Please configure your OpenAI API key in the .env file")
    st.stop()

# PAGE CONFIG

st.set_page_config(
    page_title="IMDB Movie Chat",
    layout="centered"
)

# HELPER FUNCTIONS


def create_new_thread() -> str:
    """Generate new unique thread ID"""
    return f"thread-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def get_thread_display_name(thread_id: str) -> str:
    """Get display name for thread"""
    preview = get_thread_first_message(thread_id)
    if preview and preview != thread_id:
        return preview
    return f"Chat {thread_id[-8:]}"


def get_thread_timestamp(thread_id: str) -> str:
    """Extract timestamp from thread ID for sorting"""
    try:
        # Thread ID format: thread-{uuid}-{timestamp}
        parts = thread_id.split('-')
        if len(parts) >= 3:
            # Last part is the timestamp
            return parts[-1]
        return thread_id
    except:
        return thread_id

# SESSION STATE INITIALIZATION


# Initialize current thread
if 'current_thread' not in st.session_state:
    st.session_state['current_thread'] = create_new_thread()

# Initialize message cache per thread
if 'thread_messages' not in st.session_state:
    st.session_state['thread_messages'] = {}

# Get current thread ID
current_thread = st.session_state['current_thread']

# Initialize messages for current thread if not exists
if current_thread not in st.session_state['thread_messages']:
    # Try loading from checkpointer first
    loaded_messages = get_thread_messages(current_thread)
    st.session_state['thread_messages'][current_thread] = loaded_messages


# SIDEBAR


with st.sidebar:
    st.header("IMDB Movie Chat")
    st.markdown(f"**Model:** {LLM_MODEL}")
    st.markdown("---")

    # New Chat Button
    if st.button("+ New Chat", use_container_width=True):
        new_thread = create_new_thread()
        st.session_state['current_thread'] = new_thread
        st.session_state['thread_messages'][new_thread] = []
        st.rerun()

    st.markdown("---")
    st.subheader("Chat History")

    # List all threads
    all_threads = list_all_threads()

    # Combine existing threads with current session threads
    session_threads = list(st.session_state['thread_messages'].keys())
    all_thread_ids = list(set(all_threads + session_threads))

    if all_thread_ids:
        # Sort by timestamp extracted from thread ID (most recent first)
        all_thread_ids = sorted(
            all_thread_ids,
            key=get_thread_timestamp,
            reverse=True
        )

        for thread_id in all_thread_ids[:10]:  # Show max 10 threads
            display_name = get_thread_display_name(thread_id)
            # If display name is too generic, add a visual separator
            if display_name.startswith("Chat"):
                display_name = f"{display_name}"
            else:
                display_name = f"{display_name[:40]}..." if len(
                    display_name) > 40 else f" {display_name}"

            # Create columns for thread button and delete button
            col1, col2 = st.columns([4, 1])

            with col1:
                # Highlight current thread
                if thread_id == current_thread:
                    st.success(f"{display_name}")
                else:
                    if st.button(display_name, key=f"thread_{thread_id}", use_container_width=True):
                        st.session_state['current_thread'] = thread_id
                        # Load messages from checkpointer
                        loaded_messages = get_thread_messages(thread_id)
                        st.session_state['thread_messages'][thread_id] = loaded_messages
                        st.rerun()

            with col2:
                # Only show delete button if not current thread
                if thread_id != current_thread:
                    if st.button("X", key=f"delete_{thread_id}"):
                        # Delete from checkpointer
                        delete_thread(thread_id)
                        # Delete from session state
                        if thread_id in st.session_state['thread_messages']:
                            del st.session_state['thread_messages'][thread_id]
                        st.rerun()
    else:
        st.caption("No chat history yet")

    st.markdown("---")
    st.caption("Ask me anything about the IMDB Top 1000 movies!")

# MAIN CHAT INTERFACE

st.title("IMDB Movie Chat")

# Get messages for current thread
messages = st.session_state['thread_messages'][current_thread]

# Display conversation history
for message in messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
user_input = st.chat_input("Ask about movies...")

if user_input:
    # Add user message to history
    messages.append({
        'role': 'user',
        'content': user_input
    })

    # Display user message
    with st.chat_message('user'):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response
        try:
            with st.spinner(" Thinking..."):
                for event in query_agent_stream(user_input, current_thread):
                    if 'final_response' in event and event['final_response']:
                        full_response = event['final_response']
                        response_placeholder.markdown(full_response)

            # If we got a response, show it
            if full_response:
                response_placeholder.markdown(full_response)
                messages.append({
                    'role': 'assistant',
                    'content': full_response
                })
            else:
                error_msg = "I apologize, but I couldn't generate a response. Please try again."
                response_placeholder.error(error_msg)
                messages.append({
                    'role': 'assistant',
                    'content': error_msg
                })

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            response_placeholder.error(error_msg)
            messages.append({
                'role': 'assistant',
                'content': error_msg
            })

    # Update session state
    st.session_state['thread_messages'][current_thread] = messages


# FOOTER


st.markdown("---")
