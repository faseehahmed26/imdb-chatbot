"""
Streamlit App for IMDB Conversational Agent
ChatGPT-style interface with streaming responses and multi-thread support
"""
import config
import streamlit as st
import uuid
from datetime import datetime
from agents import query_agent_stream, list_all_threads, get_thread_first_message
from config import DUCKDB_PATH, CHROMA_PATH, OPENAI_API_KEY, LLM_MODEL

# =============================================================================
# STARTUP CHECK
# =============================================================================

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

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="IMDB Movie Chat",
    page_icon="🎬",
    layout="centered"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_new_thread() -> str:
    """Generate new unique thread ID"""
    return f"thread-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def get_thread_display_name(thread_id: str) -> str:
    """Get display name for thread"""
    preview = get_thread_first_message(thread_id)
    if preview and preview != thread_id:
        return preview
    return f"Chat {thread_id[-8:]}"

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================


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
    st.session_state['thread_messages'][current_thread] = []

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🎬 IMDB Movie Chat")
    st.markdown(f"**Model:** {LLM_MODEL}")
    st.markdown("---")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
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
        # Sort threads (most recent first)
        all_thread_ids = sorted(all_thread_ids, reverse=True)

        for thread_id in all_thread_ids[:10]:  # Show max 10 threads
            display_name = get_thread_display_name(thread_id)

            # Highlight current thread
            if thread_id == current_thread:
                st.success(f"💬 {display_name}")
            else:
                if st.button(display_name, key=f"thread_{thread_id}", use_container_width=True):
                    st.session_state['current_thread'] = thread_id
                    # Initialize messages if not exists
                    if thread_id not in st.session_state['thread_messages']:
                        st.session_state['thread_messages'][thread_id] = []
                    st.rerun()
    else:
        st.caption("No chat history yet")

    st.markdown("---")
    st.caption("Ask me anything about the IMDB Top 1000 movies!")

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

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

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
