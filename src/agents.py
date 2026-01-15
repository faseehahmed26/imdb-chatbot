"""
Agents Module
Implements the 4-agent LangGraph workflow for IMDB queries
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import json
import re
import sqlite3
from pathlib import Path

from config import OPENAI_API_KEY, LLM_MODEL
from tools import get_duckdb_tool, get_chromadb_tool
from prompts import (
    format_router_prompt,
    format_structured_query_prompt,
    format_rag_query_prompt,
    format_synthesizer_prompt,
    ERROR_NO_RESULTS,
    ERROR_SQL_FAILED,
    ERROR_SEMANTIC_FAILED
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State passed between agents in the workflow"""
    query: str
    query_type: Optional[Literal["STRUCTURED", "SEMANTIC", "HYBRID"]]
    routing_reasoning: Optional[str]

    # Structured query results
    sql_query: Optional[str]
    sql_results: Optional[Dict[str, Any]]
    sql_error: Optional[str]

    # Semantic search results
    semantic_query: Optional[str]
    semantic_results: Optional[Dict[str, Any]]
    semantic_error: Optional[str]

    # Final output
    final_response: str
    error: Optional[str]


# =============================================================================
# INITIALIZE LLM
# =============================================================================

def get_llm():
    """Get configured LLM instance"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )


# =============================================================================
# ROUTER AGENT
# =============================================================================

def router_agent(state: AgentState) -> AgentState:
    """
    Route query to appropriate agent(s)
    Classifies query as STRUCTURED, SEMANTIC, or HYBRID
    """
    query = state["query"]

    try:
        llm = get_llm()
        prompts = format_router_prompt(query)

        messages = [
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]

        response = llm.invoke(messages)
        content = response.content

        # Extract query type from response
        query_type = None
        if "STRUCTURED" in content.upper():
            query_type = "STRUCTURED"
        elif "SEMANTIC" in content.upper():
            query_type = "SEMANTIC"
        elif "HYBRID" in content.upper():
            query_type = "HYBRID"
        else:
            # Default to STRUCTURED if unclear
            query_type = "STRUCTURED"

        state["query_type"] = query_type
        state["routing_reasoning"] = content

        print(f"\n[ROUTER] Classified as: {query_type}")
        print(f"[ROUTER] Reasoning: {content[:200]}...")

    except Exception as e:
        print(f"[ROUTER ERROR] {e}")
        state["query_type"] = "STRUCTURED"  # Default fallback
        state["error"] = f"Router error: {e}"

    return state


# =============================================================================
# STRUCTURED QUERY AGENT
# =============================================================================

def structured_query_agent(state: AgentState) -> AgentState:
    """
    Generate and execute SQL queries for structured data
    """
    query = state["query"]

    try:
        # Get database tool and schema
        db_tool = get_duckdb_tool()
        schema = db_tool.get_schema()

        # Generate SQL query
        llm = get_llm()
        prompts = format_structured_query_prompt(query, schema)

        messages = [
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]

        response = llm.invoke(messages)
        sql_query = response.content.strip()

        # Extract SQL from code blocks if present
        sql_match = re.search(r'```sql\n(.*?)\n```', sql_query, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1)
        elif '```' in sql_query:
            # Remove any code block markers
            sql_query = sql_query.replace(
                '```sql', '').replace('```', '').strip()

        state["sql_query"] = sql_query
        print(f"\n[STRUCTURED] Generated SQL:\n{sql_query}")

        # Execute query
        result = db_tool.execute_query(sql_query)

        if result["success"]:
            state["sql_results"] = result
            print(f"[STRUCTURED] Found {result['row_count']} results")
        else:
            state["sql_error"] = result["error"]
            print(f"[STRUCTURED ERROR] {result['error']}")

    except Exception as e:
        print(f"[STRUCTURED ERROR] {e}")
        state["sql_error"] = str(e)

    return state


# =============================================================================
# RAG AGENT
# =============================================================================

def rag_agent(state: AgentState) -> AgentState:
    """
    Perform semantic search on movie overviews
    """
    query = state["query"]

    try:
        # Get ChromaDB tool
        chroma_tool = get_chromadb_tool()

        # Generate semantic search query
        llm = get_llm()
        prompts = format_rag_query_prompt(query)

        messages = [
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]

        response = llm.invoke(messages)
        search_strategy = response.content

        # Extract search query (simplified - use full response as query)
        # In production, would parse the structured output more carefully
        search_query = query  # Use original query for now

        state["semantic_query"] = search_query
        print(f"\n[RAG] Search strategy:\n{search_strategy[:200]}...")

        # For HYBRID queries, check if we have SQL results to filter by
        where_filter = None
        if state["query_type"] == "HYBRID" and state.get("sql_results"):
            # Could implement metadata filtering here based on SQL results
            # For now, just do semantic search
            pass

        # Perform search
        result = chroma_tool.search(
            query=search_query,
            n_results=10,
            where_filter=where_filter
        )

        if result["success"]:
            state["semantic_results"] = result
            print(f"[RAG] Found {result['count']} results")
        else:
            state["semantic_error"] = result["error"]
            print(f"[RAG ERROR] {result['error']}")

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        state["semantic_error"] = str(e)

    return state


# =============================================================================
# SYNTHESIZER AGENT
# =============================================================================

def synthesizer_agent(state: AgentState) -> AgentState:
    """
    Synthesize results from structured and/or semantic queries
    Generate final conversational response
    """
    query = state["query"]

    try:
        # Prepare results summary
        sql_results_str = "No structured results"
        if state.get("sql_results") and state["sql_results"].get("success"):
            data = state["sql_results"]["data"]
            if data:
                sql_results_str = f"Found {len(data)} results:\n{json.dumps(data[:10], indent=2)}"
            else:
                sql_results_str = "Query executed successfully but returned no results"
        elif state.get("sql_error"):
            sql_results_str = f"SQL Error: {state['sql_error']}"

        semantic_results_str = "No semantic search results"
        if state.get("semantic_results") and state["semantic_results"].get("success"):
            movies = state["semantic_results"]["movies"]
            if movies:
                semantic_results_str = f"Found {len(movies)} relevant movies:\n"
                for i, movie in enumerate(movies[:5], 1):
                    meta = movie['metadata']
                    semantic_results_str += f"\n{i}. {meta.get('Series_Title', 'Unknown')} ({meta.get('Released_Year', 'N/A')})\n"
                    semantic_results_str += f"   Overview: {meta.get('Overview', 'N/A')[:150]}...\n"
            else:
                semantic_results_str = "Search completed but found no relevant results"
        elif state.get("semantic_error"):
            semantic_results_str = f"Semantic Search Error: {state['semantic_error']}"

        # Generate response
        llm = get_llm()
        prompts = format_synthesizer_prompt(
            query, sql_results_str, semantic_results_str)

        messages = [
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]

        response = llm.invoke(messages)
        final_response = response.content

        state["final_response"] = final_response
        print(
            f"\n[SYNTHESIZER] Generated response ({len(final_response)} chars)")

    except Exception as e:
        print(f"[SYNTHESIZER ERROR] {e}")
        state["final_response"] = f"I encountered an error generating the response: {e}"
        state["error"] = str(e)

    return state


# =============================================================================
# CHECKPOINTER SETUP
# =============================================================================

# Create SQLite checkpointer for conversation memory
_checkpointer = None


def get_checkpointer():
    """Get or create SQLite checkpointer instance"""
    global _checkpointer
    if _checkpointer is None:
        # Create data directory if it doesn't exist
        db_path = Path("data/imdb_chatbot.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create SQLite connection with thread safety
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer = SqliteSaver(conn=conn)
    return _checkpointer


# =============================================================================
# LANGGRAPH WORKFLOW
# =============================================================================

def route_based_on_type(state: AgentState) -> List[str]:
    """
    Conditional routing based on query type
    Returns list of next nodes to execute
    """
    query_type = state.get("query_type", "STRUCTURED")

    if query_type == "STRUCTURED":
        return ["structured_query"]
    elif query_type == "SEMANTIC":
        return ["semantic_query"]
    else:  # HYBRID
        return ["structured_query", "semantic_query"]


def create_workflow() -> StateGraph:
    """
    Create and compile the LangGraph workflow with checkpointer
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route_query", router_agent)
    workflow.add_node("structured_query", structured_query_agent)
    workflow.add_node("semantic_query", rag_agent)
    workflow.add_node("synthesize", synthesizer_agent)

    # Set entry point
    workflow.set_entry_point("route_query")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "route_query",
        route_based_on_type,
        {
            "structured_query": "structured_query",
            "semantic_query": "semantic_query"
        }
    )

    # Both query types lead to synthesizer
    workflow.add_edge("structured_query", "synthesize")
    workflow.add_edge("semantic_query", "synthesize")

    # Synthesizer is the end
    workflow.add_edge("synthesize", END)

    # Compile with checkpointer for conversation memory
    return workflow.compile(checkpointer=get_checkpointer())


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Create workflow instance
_workflow = None


def get_workflow():
    """Get or create workflow instance"""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow


def query_agent(query: str) -> Dict[str, Any]:
    """
    Main interface to query the agent system

    Args:
        query: User's natural language query

    Returns:
        Dict containing the final response and intermediate results
    """
    workflow = get_workflow()

    # Initialize state
    initial_state = {
        "query": query,
        "query_type": None,
        "routing_reasoning": None,
        "sql_query": None,
        "sql_results": None,
        "sql_error": None,
        "semantic_query": None,
        "semantic_results": None,
        "semantic_error": None,
        "final_response": "",
        "error": None
    }

    # Execute workflow
    try:
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")

        result = workflow.invoke(initial_state)

        return {
            "success": True,
            "response": result["final_response"],
            "query_type": result.get("query_type"),
            "sql_query": result.get("sql_query"),
            "sql_results": result.get("sql_results"),
            "semantic_results": result.get("semantic_results"),
            "routing_reasoning": result.get("routing_reasoning"),
            "error": result.get("error")
        }

    except Exception as e:
        print(f"\n[WORKFLOW ERROR] {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "response": f"I encountered an error processing your query: {e}",
            "error": str(e)
        }


def query_agent_stream(query: str, thread_id: str = "default"):
    """
    Stream agent responses with conversation memory

    Args:
        query: User's natural language query
        thread_id: Unique identifier for conversation thread

    Yields:
        Dict containing intermediate state updates
    """
    workflow = get_workflow()
    config = {'configurable': {'thread_id': thread_id}}

    # Initialize state
    initial_state = {
        "query": query,
        "query_type": None,
        "routing_reasoning": None,
        "sql_query": None,
        "sql_results": None,
        "sql_error": None,
        "semantic_query": None,
        "semantic_results": None,
        "semantic_error": None,
        "final_response": "",
        "error": None
    }

    # Stream workflow execution
    try:
        for event in workflow.stream(initial_state, config=config, stream_mode="values"):
            yield event
    except Exception as e:
        print(f"\n[WORKFLOW STREAM ERROR] {e}")
        import traceback
        traceback.print_exc()
        yield {
            "final_response": f"I encountered an error processing your query: {e}",
            "error": str(e)
        }


# =============================================================================
# THREAD MANAGEMENT
# =============================================================================

def list_all_threads() -> List[str]:
    """List all conversation threads"""
    try:
        all_threads = set()
        checkpointer = get_checkpointer()
        for cp in checkpointer.list(None):
            all_threads.add(cp.config['configurable']['thread_id'])
        return sorted(list(all_threads), reverse=True)  # Most recent first
    except Exception as e:
        print(f"Error listing threads: {e}")
        return []


def get_thread_messages(thread_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all messages for a specific thread

    Args:
        thread_id: Thread identifier

    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    try:
        workflow = get_workflow()
        config = {'configurable': {'thread_id': thread_id}}
        state = workflow.get_state(config)

        if state and state.values:
            # Extract conversation history from state
            messages = []
            # The state contains query/response pairs
            # We'll reconstruct a simple message history
            if state.values.get('query'):
                messages.append({
                    'role': 'user',
                    'content': state.values['query']
                })
            if state.values.get('final_response'):
                messages.append({
                    'role': 'assistant',
                    'content': state.values['final_response']
                })
            return messages
        return []
    except Exception as e:
        print(f"Error getting thread messages: {e}")
        return []


def get_thread_first_message(thread_id: str) -> str:
    """Get the first user message from a thread for preview"""
    messages = get_thread_messages(thread_id)
    if messages and len(messages) > 0:
        first_msg = messages[0]['content']
        # Truncate to 50 characters
        return first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
    return thread_id
