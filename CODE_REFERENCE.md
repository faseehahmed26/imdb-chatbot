# Code Reference

Detailed explanation of each file and key functions in the IMDB Agent project.

---

## `src/config.py`

**Purpose**: Central configuration and environment setup

### Key Variables
- `PROJECT_ROOT`: Base directory path
- `DATA_DIR`: Where CSV and DuckDB files are stored
- `VECTORSTORE_DIR`: ChromaDB persistence location
- `DUCKDB_PATH`: Path to `imdb.duckdb` file
- `CSV_PATH`: Path to source CSV file
- `OPENAI_API_KEY`: OpenAI API key from `.env`
- `LLM_MODEL`: Model name (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)

### What It Does
- Loads environment variables from `.env` file
- Sets up LangSmith tracing if configured
- Creates required directories
- Validates configuration

---

## `src/data_setup.py`

**Purpose**: One-time data pipeline to create DuckDB and ChromaDB

### Key Functions

#### `load_and_clean_data()`
- Loads CSV using pandas
- Cleans data:
  - Converts `Gross` from string "28,341,469" → float 28341469.0
  - Converts `Runtime` from "142 min" → integer 142
  - Converts `Released_Year` to integer
  - Handles missing values
- Returns cleaned DataFrame

#### `setup_duckdb(df)`
- Creates DuckDB database file
- Creates `imdb` table from DataFrame
- Adds indexes on commonly queried columns (year, rating, director)
- Returns success status

#### `setup_chromadb(df)`
- Initializes ChromaDB persistent client
- For each movie, generates embedding using OpenAI API
- Embedding text format: `"{Title} | {Genre} | {Director} | {Overview}"`
- Stores all metadata (ratings, year, cast, etc.)
- Processes in batches of 100 for efficiency
- Returns success status

#### `main()`
- Orchestrates entire setup process
- Calls all three functions in sequence
- Handles errors and displays progress

### Usage
```bash
python -m src.data_setup
```

---

## `src/tools.py`

**Purpose**: Wrapper classes for DuckDB and ChromaDB operations

### Class: `DuckDBTool`

#### Methods

**`__init__(db_path)`**
- Connects to DuckDB database
- Validates database exists

**`execute_query(sql: str) → dict`**
- Executes SQL query
- Returns: `{"success": True, "data": [...], "row_count": N}` on success
- Returns: `{"success": False, "error": "..."}` on failure

**`get_schema() → str`**
- Returns formatted schema with column names, types, descriptions
- Includes sample data
- Used in structured query prompts

**`get_table_stats() → dict`**
- Returns database statistics (total movies, year range, etc.)

### Class: `ChromaDBTool`

#### Methods

**`__init__(persist_dir, collection_name)`**
- Connects to ChromaDB persistent store
- Gets collection by name

**`search(query: str, n_results: int, where_filter: dict) → dict`**
- Performs semantic vector search
- Optional metadata filtering (e.g., genre, year)
- Returns: `{"success": True, "movies": [...], "count": N}`

**`search_with_structured_filter(query: str, movie_ids: list, n_results: int)`**
- Searches within a subset of movies
- Used for hybrid queries (SQL + semantic)

**`get_collection_stats() → dict`**
- Returns collection info (total documents, sample titles)

### Singleton Functions

**`get_duckdb_tool()`** - Returns shared DuckDB instance

**`get_chromadb_tool()`** - Returns shared ChromaDB instance

---

## `src/prompts.py`

**Purpose**: All LLM prompt templates

### Prompt Constants

#### Router Prompts
**`ROUTER_SYSTEM_PROMPT`**
- Instructs LLM to classify queries as STRUCTURED/SEMANTIC/HYBRID
- Contains examples for each type

**`ROUTER_USER_PROMPT`**
- Template: `"User Query: {query}\n\nClassify this query..."`

#### Structured Query Prompts
**`STRUCTURED_QUERY_SYSTEM_PROMPT`**
- Contains database schema
- SQL generation instructions
- Common query patterns and examples
- Chain-of-thought prompting

**`STRUCTURED_QUERY_USER_PROMPT`**
- Template asking for SQL query

#### RAG Prompts
**`RAG_QUERY_SYSTEM_PROMPT`**
- Instructions for semantic search
- How to extract concepts and formulate queries
- Examples of metadata filtering

**`RAG_QUERY_USER_PROMPT`**
- Template for semantic search strategy

#### Synthesizer Prompts
**`SYNTHESIZER_SYSTEM_PROMPT`**
- How to format responses
- Conversational tone guidelines
- Markdown formatting rules

**`SYNTHESIZER_USER_PROMPT`**
- Template with query + results

### Helper Functions

**`format_router_prompt(query)`** - Returns formatted router prompts

**`format_structured_query_prompt(query, schema)`** - Returns SQL generation prompts

**`format_rag_query_prompt(query)`** - Returns semantic search prompts

**`format_synthesizer_prompt(query, sql_results, semantic_results)`** - Returns synthesis prompts

---

## `src/agents.py`

**Purpose**: Core agent system and LangGraph workflow

### State Definition

**`AgentState` (TypedDict)**
- `query`: User's question
- `query_type`: STRUCTURED/SEMANTIC/HYBRID
- `routing_reasoning`: Why this classification
- `sql_query`: Generated SQL (if applicable)
- `sql_results`: Query results (if applicable)
- `semantic_query`: Search query (if applicable)
- `semantic_results`: Search results (if applicable)
- `final_response`: Final answer
- `error`: Any errors

### Agent Functions

#### `router_agent(state: AgentState) → AgentState`
**What it does:**
1. Takes user query from state
2. Calls LLM with router prompts
3. Classifies as STRUCTURED/SEMANTIC/HYBRID
4. Updates state with classification and reasoning
5. Returns updated state

#### `structured_query_agent(state: AgentState) → AgentState`
**What it does:**
1. Gets database schema from DuckDBTool
2. Calls LLM to generate SQL query
3. Executes SQL using DuckDBTool
4. Updates state with query and results
5. Handles errors (SQL syntax, execution failures)

#### `rag_agent(state: AgentState) → AgentState`
**What it does:**
1. Calls LLM to formulate semantic search
2. Executes search using ChromaDBTool
3. Updates state with search query and results
4. Handles errors (no results, API failures)

#### `synthesizer_agent(state: AgentState) → AgentState`
**What it does:**
1. Combines SQL results and/or semantic results
2. Formats for LLM prompt
3. Calls LLM to generate conversational response
4. Updates state with final response

### Workflow Functions

#### `route_based_on_type(state) → List[str]`
**Conditional routing logic:**
- If STRUCTURED → returns `["structured_query"]`
- If SEMANTIC → returns `["semantic_query"]`
- If HYBRID → returns `["structured_query", "semantic_query"]` (both)

#### `create_workflow() → StateGraph`
**Creates LangGraph workflow:**
1. Adds all 4 agent nodes
2. Sets entry point to `route_query`
3. Adds conditional edges based on query type
4. Both query agents lead to synthesizer
5. Synthesizer leads to END
6. Compiles and returns workflow

#### `query_agent(query: str) → dict`
**Main entry point:**
- Initializes state with user query
- Invokes workflow
- Returns result dictionary with:
  - `success`: bool
  - `response`: final answer
  - `query_type`, `sql_query`, `sql_results`, `semantic_results`
  - `error`: if any

---

## `src/app.py`

**Purpose**: Streamlit web UI

### Key Components

#### Configuration Sidebar
- Shows API key status
- Displays current model
- Sample question buttons
- Clear chat button
- About section

#### Main Chat Interface
- Message history display
- Chat input box
- Expandable agent details:
  - Query routing decision
  - SQL query + results table
  - Semantic search results

#### Message Handling
**When user sends a query:**
1. Validates API key
2. Adds to message history
3. Calls `query_agent(user_input)`
4. Displays response
5. Shows intermediate results in expanders
6. Handles errors gracefully

#### Startup Checks
- Verifies OpenAI API key configured
- Checks if DuckDB and ChromaDB exist
- Shows setup instructions if missing

### Usage
```bash
streamlit run src/app.py
```

---

## `src/telegram_bot.py`

**Purpose**: Telegram bot integration

### Command Handlers

#### `start_command(update, context)`
- Sends welcome message
- Lists example questions
- Shows available commands

#### `help_command(update, context)`
- Explains what the bot can do
- Shows query types
- Provides tips

#### `examples_command(update, context)`
- Lists all 9 test questions

### Message Handler

#### `handle_message(update, context)`
**What it does:**
1. Gets user message text
2. Validates API key
3. Shows typing indicator
4. Calls `query_agent(user_query)`
5. Formats response for Telegram
6. Splits long messages (4096 char limit)
7. Sends query type as follow-up
8. Handles errors

### Error Handler

#### `error_handler(update, context)`
- Logs errors
- Sends user-friendly error message

### Main Function

#### `main()`
- Validates configuration (API keys, databases)
- Creates Telegram Application
- Registers all handlers
- Starts polling

### Usage
```bash
python -m src.telegram_bot
```

---

## `src/test_queries.py`

**Purpose**: Test suite for all 9 assignment questions

### Test Data

**`TEST_QUESTIONS`** - List of 9 test cases with:
- `id`: Question number
- `query`: The question text
- `expected_type`: Expected classification (STRUCTURED/SEMANTIC/HYBRID)

### Main Function

#### `test_all_queries()`
**What it does:**
1. Validates database setup
2. Loops through all test questions
3. Calls `query_agent()` for each
4. Validates:
   - Success status
   - Query type matches expected
   - SQL/semantic queries executed
   - Response generated
5. Prints detailed results for each test
6. Displays summary statistics

### Usage
```bash
python -m src.test_queries
```

---

## How It All Works Together

### Example: "Top 5 movies of 2019"

1. **User** types query in Streamlit
2. **app.py** calls `query_agent(query)`
3. **agents.py** `query_agent()`:
   - Creates initial state with query
   - Invokes LangGraph workflow
4. **Router Agent**:
   - Calls LLM with router prompts
   - Classifies as STRUCTURED
   - Updates state
5. **Workflow** routes to `structured_query` node
6. **Structured Query Agent**:
   - Gets schema from `DuckDBTool`
   - Calls LLM to generate SQL
   - SQL: `SELECT ... WHERE Released_Year = 2019 ORDER BY ... LIMIT 5`
   - Executes via `DuckDBTool.execute_query()`
   - Updates state with results
7. **Workflow** routes to `synthesize` node
8. **Synthesizer Agent**:
   - Formats SQL results for LLM
   - Calls LLM to generate conversational response
   - Updates state with final response
9. **Workflow** returns to `query_agent()`
10. **app.py** displays:
    - Final response
    - Routing decision (STRUCTURED)
    - SQL query in expander
    - Results table

---

## Key Design Decisions

### Why Separate Tools?
- **Modularity**: Easy to swap DuckDB for PostgreSQL
- **Testing**: Can test tools independently
- **Reusability**: Same tools used by all agents

### Why TypedDict for State?
- **Type safety**: IDE autocomplete and type checking
- **Clear contract**: Each agent knows what data is available
- **LangGraph requirement**: StateGraph needs typed state

### Why Separate Prompts File?
- **Easy iteration**: Change prompts without touching agent logic
- **A/B testing**: Can experiment with different prompts
- **Maintainability**: All prompts in one place

### Why Multiple Agents Instead of One?
- **Separation of concerns**: Each agent has one job
- **Debugging**: Easy to see which agent failed
- **Flexibility**: Can skip agents based on query type
- **Monitoring**: LangSmith traces show each agent's work

---

## Extending the System

### Adding a New Query Type
1. Update `AgentState` to include new query type
2. Update router prompts to recognize it
3. Create new agent function
4. Update `route_based_on_type()` logic
5. Add new node to workflow

### Adding a New Data Source
1. Create new tool in `tools.py`
2. Update `data_setup.py` to ingest data
3. Create new agent or extend existing
4. Update prompts with new schema

### Adding Voice Capabilities
1. Create `src/voice.py` with Whisper/TTS functions
2. Update `telegram_bot.py` to accept voice messages
3. Convert audio → text → agent → text → audio

---

## Common Patterns

### Error Handling Pattern
```python
try:
    result = tool.execute()
    if result['success']:
        state['field'] = result['data']
    else:
        state['field_error'] = result['error']
except Exception as e:
    state['field_error'] = str(e)
```

### LLM Call Pattern
```python
llm = get_llm()
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
]
response = llm.invoke(messages)
content = response.content
```

### Tool Singleton Pattern
```python
_tool_instance = None

def get_tool():
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = Tool()
    return _tool_instance
```

---

**This reference covers all major components. Check inline code comments for implementation details!**
