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
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessageState


# ============================================================================
# Configuration
# ============================================================================

DB_URL = os.environ.get("DB_URL", "sqlite:///:memory:")
INCLUDE_TABLES: Optional[List[str]] = None
DATA_DICTIONARY_PATH = os.environ.get("DATA_DICTIONARY_PATH", "data_dictionary.json")
TOP_K_DEFAULT = 20

# Cache for repeated queries and large schema support
_DATA_DICTIONARY_CACHE: Dict[str, Any] = {}
_SCHEMA_INDEX_CACHE: Optional[Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]] = None

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
    global _DATA_DICTIONARY_CACHE
    if _DATA_DICTIONARY_CACHE.get("path") == path:
        return _DATA_DICTIONARY_CACHE.get("data", {})

    path_obj = Path(path)
    if not path_obj.exists():
        _DATA_DICTIONARY_CACHE = {"path": path, "data": {}}
        return {}

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
            _DATA_DICTIONARY_CACHE = {"path": path, "data": data}
            return data
    except Exception:
        _DATA_DICTIONARY_CACHE = {"path": path, "data": {}}
        return {}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 1]


def _build_schema_index(data_dict: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build keyword-to-table/column index for relevance scoring."""
    table_index: Dict[str, Set[str]] = {}
    column_index: Dict[str, Set[str]] = {}

    for table_name, info in data_dict.items():
        for token in _tokenize(table_name):
            table_index.setdefault(token, set()).add(table_name)

        for text_field in (info.get("table_description") or "", info.get("description") or ""):
            for token in _tokenize(text_field):
                table_index.setdefault(token, set()).add(table_name)

        for col in info.get("columns", []):
            col_name = col.get("name", "")
            col_desc = col.get("col_description", "") or col.get("type", "")
            sample_values = col.get("_prompt_samples", []) or []

            for token in _tokenize(col_name):
                column_index.setdefault(token, set()).add(table_name)
            for token in _tokenize(col_desc):
                column_index.setdefault(token, set()).add(table_name)
            for sample in sample_values:
                for token in _tokenize(str(sample)):
                    column_index.setdefault(token, set()).add(table_name)

    return table_index, column_index


def _select_candidate_tables(
    question: str,
    data_dict: Dict[str, Any],
    max_tables: int = 5,
) -> List[str]:
    """Pick relevant tables based on question keywords, descriptions, and sample values."""
    if not data_dict:
        return []

    question_tokens = set(_tokenize(question))
    table_index, column_index = _build_schema_index(data_dict)

    scores: Dict[str, int] = {}
    for tok in question_tokens:
        # Table name or description matches are the strongest signal
        for tbl in table_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 6
        # Column names, descriptions, and sample values are also important
        for tbl in column_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 4

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return []

    selected = [t for t, score in ranked if score > 0][:max_tables]
    return selected


def _format_schema_for_prompt(data_dict: Dict[str, Any], tables: List[str]) -> str:
    """Format selected schema into a full prompt block for the LLM."""
    if not tables:
        return ""

    blocks: List[str] = [
        "Known schema for selected tables:",
        "Use only these tables and columns. Do not invent table names or columns.",
    ]

    for table in tables:
        info = data_dict.get(table, {})
        table_description = info.get("table_description") or info.get("description") or ""
        blocks.append(f"{table}:")
        if table_description:
            blocks.append(f"  description: {table_description}")
        blocks.append("  columns:")

        for col in info.get("columns", []):
            name = col.get("name")
            col_type = col.get("type") or "UNKNOWN"
            col_desc = col.get("col_description") or ""
            if col_desc:
                blocks.append(f"    - {name} ({col_type}): {col_desc}")
            else:
                blocks.append(f"    - {name} ({col_type})")

        foreign_keys = info.get("foreign_keys") or []
        if foreign_keys:
            fk_lines = [
                f"    - {fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}"
                for fk in foreign_keys
                if fk.get('column') and fk.get('references_table')
            ]
            if fk_lines:
                blocks.append("  foreign_keys:")
                blocks.extend(fk_lines)

    return "\n".join(blocks)


def _clean_query(query: str) -> str:
    """Clean up SQL query: remove extra whitespace and normalize newlines."""
    # Replace multiple newlines with single space, preserve structure
    query = re.sub(r'\s+', ' ', query.strip())
    return query


def _apply_result_limit(query: str, top_k: int) -> str:
    """Ensure query returns at most top_k rows (only add if not present)."""
    query = _clean_query(query)
    # Remove trailing semicolon for SQLAlchemy/database compatibility
    query = query.rstrip(";")
    lc = query.lower()
    # Don't add limit if one already exists
    if "limit" in lc or "fetch first" in lc or "rownum" in lc or "offset" in lc:
        return query
    # Only add limit if user isn't asking for all rows explicitly
    return query + f" LIMIT {top_k}"


# ============================================================================
# System Prompts
# ============================================================================

GENERATE_QUERY_SYSTEM_PROMPT = """
You are an expert SQL agent designed to interact with a large database (1500+ tables, billions of rows).
Given an input question, create a syntactically correct {dialect} query to run.

**IMPORTANT RULES:**
1. Use ONLY the tables and columns listed in the schema below.
2. Always return results using LIMIT {top_k} unless the user asks for all data.
3. Select the relevant columns needed to answer the question. If the user asks for "all data from table X", use SELECT * FROM X.
4. Order results by a relevant column to return the most interesting examples.
5. Never use JOIN with unknown tables. Only join tables explicitly mentioned in the schema.
6. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, ALTER, etc.).
7. Return ONLY the SQL query. No explanation or markdown formatting.
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
- Referencing only columns that exist in the schema used to generate the query

If there are any mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
Only return the SQL query, nothing else.
""".strip()


def _is_sql_query(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith("select") or normalized.startswith("with") or " from " in normalized


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
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            user_question = msg.content.strip()
            break

    if not user_question:
        return {
            "messages": messages + [
                AIMessage(content="I need a question to generate a SQL query.")
            ]
        }

    data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)
    candidate_tables = _select_candidate_tables(user_question, data_dict, max_tables=5)

    if not candidate_tables:
        clarification = (
            "I cannot safely generate a query because the question does not match any tables or columns "
            "in the provided data dictionary. Please clarify your request or provide more context."
        )
        return {"messages": messages + [AIMessage(content=clarification)]}

    schema_hint = _format_schema_for_prompt(data_dict, candidate_tables)
    system_prompt = GENERATE_QUERY_SYSTEM_PROMPT.format(dialect=dialect, top_k=TOP_K_DEFAULT)
    system_prompt += (
        "\n\nUse only the schema shown below. Do not invent new table names or columns. "
        "If the question cannot be answered from this schema, ask for clarification instead of guessing."
    )
    system_prompt += "\n\n" + schema_hint

    reasoning_message = ToolMessage(
        content=(
            f"Schema reasoning:\n"
            f"- Selected tables: {', '.join(candidate_tables)}\n"
            f"- Source: {DATA_DICTIONARY_PATH}\n"
            f"- Matching is based on table names, descriptions, column names, column descriptions, and sample values."
        ),
        tool_call_id="schema_selection"
    )

    system_message = {"role": "system", "content": system_prompt}
    response = llm.invoke([system_message] + messages)
    query = getattr(response, "content", str(response)).strip()

    if query.startswith("```sql"):
        query = query[6:]
    if query.startswith("```"):
        query = query[3:]
    if query.endswith("```"):
        query = query[:-3]

    query = _clean_query(query)
    if not _is_sql_query(query):
        return {"messages": messages + [reasoning_message, AIMessage(content=query)]}

    query = _apply_result_limit(query, TOP_K_DEFAULT)

    tool_call = {
        "name": "sql_db_query",
        "args": {"query": query},
        "id": "query_call",
        "type": "tool_call",
    }
    return {
        "messages": messages + [
            reasoning_message,
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
    
    # Clean up query: remove markdown code blocks and semicolons
    if checked_query.startswith("```sql"):
        checked_query = checked_query[6:]
    if checked_query.startswith("```"):
        checked_query = checked_query[3:]
    if checked_query.endswith("```"):
        checked_query = checked_query[:-3]
    
    # Remove semicolons and normalize whitespace
    checked_query = _clean_query(checked_query).rstrip(";")
    lc = checked_query.lower()
    if "limit" not in lc and "fetch first" not in lc and "rownum" not in lc:
        checked_query = checked_query + f" LIMIT {TOP_K_DEFAULT}"

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
    else:
        # Update the original tool call args with the cleaned query
        updated_tool_call = {
            "name": "sql_db_query",
            "args": {"query": checked_query},
            "id": tool_call.get("id", "query_call"),
            "type": "tool_call",
        }
        return {
            "messages": messages[:-1] + [
                AIMessage(
                    content=last_message.content,
                    tool_calls=[updated_tool_call]
                )
            ]
        }


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

    # Remove trailing semicolon from query before execution
    query = query.rstrip(";")

    # Execute query using the toolkit tool
    try:
        result = run_query_tool.invoke(query)
        result_str = str(result).strip()
        
        # Check if the result contains error indicators
        error_indicators = [
            "Error:",
            "error executing query",
            "SQL command not properly ended",
            "ORA-",
            "DatabaseError",
            "ProgrammingError",
            "OperationalError",
            "database error",
            "exception",
        ]
        
        is_error = any(indicator.lower() in result_str.lower() for indicator in error_indicators)
        
        if is_error:
            result_message = ToolMessage(
                content=f"Error executing query:\n\n{result_str}",
                tool_call_id=tool_call.get("id", "query_call")
            )
        else:
            result_message = ToolMessage(
                content=f"Query executed successfully.\n\nResults:\n{result_str}",
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

