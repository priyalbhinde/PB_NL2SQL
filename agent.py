"""SQL Agent with LangGraph StateGraph architecture for agent-chat-ui integration.

This agent uses a precomputed data dictionary to efficiently handle large schemas
(1500+ tables, billions of rows) by selecting only relevant tables for each query.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import load_chat_model
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from typing import Literal
from langchain.prebuilt import ToolNode
from langchain.messages import AIMessage, ToolMessage, BaseMessage
from langgraph.graph import END, START, MessageState, StateGraph


# ============================================================================
# Configuration
# ============================================================================

DB_URL = os.environ.get("DB_URL", "sqlite:///:memory:")
INCLUDE_TABLES: Optional[List[str]] = None
DATA_DICTIONARY_PATH = os.environ.get("DATA_DICTIONARY_PATH", "data_dictionary.json")
TOP_K_DEFAULT = 20

# ============================================================================
# Initialize LLM, Database, and Tools
# ============================================================================

llm: ChatOpenAI = load_chat_model()
engine = create_engine(DB_URL)
db = SQLDatabase(engine, include_tables=INCLUDE_TABLES)
dialect = getattr(db, "dialect", "sql")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
run_query_tool = next(t for t in tools if t.name == "sql_db_query")

# Create ToolNode for running queries
run_query_node = ToolNode([run_query_tool], name="run_query")


# ============================================================================
# Data Dictionary & Schema Optimization Functions
# ============================================================================

def load_data_dictionary(path: str) -> Dict[str, Any]:
    """Load (and cache) the data dictionary JSON produced by dd.py."""
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 1]


def _build_schema_index(data_dict: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build keyword-to-table/column index for relevance scoring."""
    table_index: Dict[str, Set[str]] = {}
    column_index: Dict[str, Set[str]] = {}

    for table_name, info in data_dict.items():
        lower_table = table_name.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", lower_table):
            table_index.setdefault(token, set()).add(table_name)

        for col in info.get("columns", []):
            col_name = col.get("name", "").lower()
            for token in re.findall(r"[A-Za-z0-9_]+", col_name):
                column_index.setdefault(token, set()).add(table_name)

    return table_index, column_index


def _select_candidate_tables(
    question: str,
    data_dict: Dict[str, Any],
    max_tables: int = 2,
) -> List[str]:
    """Pick relevant tables based on question keywords."""
    if not data_dict:
        return []

    question_tokens = set(_tokenize(question))
    table_index, column_index = _build_schema_index(data_dict)

    scores: Dict[str, int] = {}
    for tok in question_tokens:
        for tbl in table_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 3
        for tbl in column_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 1

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return []
    return [t for t, _ in ranked[:max_tables]]


def _format_schema_for_prompt(data_dict: Dict[str, Any], tables: List[str]) -> str:
    """Format selected schema into compact prompt block."""
    if not tables:
        return ""

    blocks: List[str] = ["Known schema (table -> columns / description):"]
    for table in tables:
        info = data_dict.get(table, {})
        cols = info.get("columns", [])
        col_lines: List[str] = []
        for col in cols:
            name = col.get("name")
            desc = col.get("col_description") or col.get("type") or ""
            col_lines.append(f"  - {name}: {desc}".strip())
        blocks.append(f"{table}:\n" + "\n".join(col_lines[:20]))
        if len(col_lines) > 20:
            blocks.append(f"  ... ({len(col_lines) - 20} more columns)")
    return "\n".join(blocks)


def _apply_result_limit(query: str, top_k: int) -> str:
    """Ensure query returns at most top_k rows."""
    lc = query.lower()
    if "limit" in lc or "fetch first" in lc or "rownum" in lc:
        return query
    return query.strip().rstrip(";") + f" LIMIT {top_k}"


# ============================================================================
# System Prompts
# ============================================================================

GENERATE_QUERY_SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run.
Unless the user specifies a specific number of examples, limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting examples.
Never query for all columns from a specific table, only the relevant ones.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".strip()


CHECK_QUERY_SYSTEM_PROMPT = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
Only return the SQL query, nothing else.
""".strip()


# ============================================================================
# Node Functions
# ============================================================================

def generate_sql_query(state: MessageState) -> Dict[str, Any]:
    """Generate SQL query from user question using schema hints.
    
    This node:
    1. Loads the data dictionary
    2. Selects relevant tables based on the question
    3. Prompts the LLM to generate a safe, focused query
    4. Applies automatic result limits
    """
    messages: List[BaseMessage] = state.get("messages", [])
    
    # Extract the latest user message
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            user_question = msg.content
            break

    if not user_question:
        return {
            "messages": messages + [
                AIMessage(content="I need a question to generate a SQL query.")
            ]
        }

    # Load data dictionary and select relevant tables
    data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)
    candidate_tables = _select_candidate_tables(user_question, data_dict, max_tables=2)
    schema_hint = _format_schema_for_prompt(data_dict, candidate_tables)

    # Build system prompt with schema hints
    system_prompt = GENERATE_QUERY_SYSTEM_PROMPT.format(dialect=dialect, top_k=TOP_K_DEFAULT)
    if schema_hint:
        system_prompt += "\n\n" + schema_hint
        system_prompt += "\n\nOnly use the above tables and columns when generating the query."

    # Generate query from LLM
    system_message = {"role": "system", "content": system_prompt}
    response = llm.invoke([system_message] + messages)
    query = getattr(response, "content", str(response)).strip()
    query = _apply_result_limit(query, TOP_K_DEFAULT)

    # Return as AIMessage with tool_call for downstream routing
    tool_call = {
        "name": "sql_db_query",
        "args": {"query": query},
        "id": "query_call",
        "type": "tool_call",
    }
    return {
        "messages": messages + [
            AIMessage(content=f"Generated query:\n```sql\n{query}\n```", tool_calls=[tool_call])
        ]
    }


def check_query_safety(state: MessageState) -> Dict[str, Any]:
    """Review and refine the generated query for safety and correctness."""
    messages: List[BaseMessage] = state.get("messages", [])

    # Extract the query from the last tool call
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": messages}

    tool_call = last_message.tool_calls[0]
    original_query = tool_call.get("args", {}).get("query", "")

    if not original_query:
        return {"messages": messages}

    # Review query with LLM
    system_prompt = CHECK_QUERY_SYSTEM_PROMPT.format(dialect=dialect)
    system_message = {"role": "system", "content": system_prompt}
    user_message = {"role": "user", "content": f"Review this query:\n{original_query}"}

    response = llm.invoke([system_message, user_message])
    checked_query = getattr(response, "content", str(response)).strip()
    checked_query = _apply_result_limit(checked_query, TOP_K_DEFAULT)

    # If the query changed, update the tool call
    if checked_query != original_query:
        updated_tool_call = {
            "name": "sql_db_query",
            "args": {"query": checked_query},
            "id": "query_call_checked",
            "type": "tool_call",
        }
        return {
            "messages": messages + [
                AIMessage(
                    content=f"Reviewed query:\n```sql\n{checked_query}\n```",
                    tool_calls=[updated_tool_call]
                )
            ]
        }

    return {"messages": messages}


def execute_query(state: MessageState) -> Dict[str, Any]:
    """Execute the SQL query and return results."""
    messages: List[BaseMessage] = state.get("messages", [])

    # Extract query from the last tool call
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": messages}

    tool_call = last_message.tool_calls[0]
    query = tool_call.get("args", {}).get("query", "")

    if not query:
        return {"messages": messages}

    # Execute query using the toolkit tool
    try:
        result = run_query_tool.invoke(query)
        result_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call.get("id", "query_call")
        )
    except Exception as e:
        result_message = ToolMessage(
            content=f"Error executing query: {str(e)}",
            tool_call_id=tool_call.get("id", "query_call")
        )

    return {"messages": messages + [result_message]}


# ============================================================================
# Router Functions
# ============================================================================

def should_check_query(state: MessageState) -> Literal["check_query", "execute_query"]:
    """Route to check_query step."""
    return "check_query"


def should_execute(state: MessageState) -> Literal["execute_query", END]:
    """Always execute after checking."""
    return "execute_query"


# ============================================================================
# Build LangGraph StateGraph
# ============================================================================

def build_agent() -> StateGraph:
    """Build and return the compiled LangGraph agent."""
    
    # Create the state graph
    builder = StateGraph(MessageState)

    # Add nodes
    builder.add_node("generate_query", generate_sql_query)
    builder.add_node("check_query", check_query_safety)
    builder.add_node("execute_query", execute_query)

    # Add edges: START -> generate_query -> check_query -> execute_query -> END
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "check_query")
    builder.add_edge("check_query", "execute_query")
    builder.add_edge("execute_query", END)

    # Compile the graph
    return builder.compile()


# Instantiate the agent
agent = build_agent()


# ============================================================================
# Main / CLI interface
# ============================================================================

def main() -> None:
    """Interactive CLI for testing the agent."""
    print(f"Connected to database: {DB_URL}")

    data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)
    if data_dict:
        tables = sorted(data_dict.keys())
        print(f"Loaded schema for {len(tables)} tables from {DATA_DICTIONARY_PATH}")
        print("Example tables:", ", ".join(tables[:10]))
    else:
        print("No data dictionary found.")

    question = input("\nEnter a natural language question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    # Invoke agent with the question
    result = agent.invoke({"messages": [AIMessage(content=question)]})

    # Print the conversation
    print("\n" + "=" * 60)
    for msg in result.get("messages", []):
        role = "Agent" if isinstance(msg, AIMessage) else "System"
        print(f"\n[{role}]:\n{msg.content[:500]}")


if __name__ == "__main__":
    main()

