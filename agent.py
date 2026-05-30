"""SQL Agent with LangGraph StateGraph architecture for agent-chat-ui integration.

This agent uses a precomputed data dictionary to efficiently handle large schemas
(1500+ tables, billions of rows) by selecting only relevant tables for each query.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import load_chat_model, get_message_text
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from typing import Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState


# ============================================================================
# Configuration
# ============================================================================

DB_URL = os.environ.get("DB_URL", "sqlite:///:memory:")
INCLUDE_TABLES: Optional[List[str]] = None
DATA_DICTIONARY_PATH = os.environ.get("DATA_DICTIONARY_PATH", "data_dictionary.json")
QUERY_EXAMPLES_PATH = os.environ.get("QUERY_EXAMPLES_PATH", "query_examples.json")
FK_JOIN_PATH = os.environ.get("FK_JOIN_PATH", "foreign_keys.json")
TOP_K_DEFAULT = 20
SCRIPT_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Cache for repeated queries and large schema support
_DATA_DICTIONARY_CACHE: Dict[str, Any] = {}
_QUERY_EXAMPLES_CACHE: Dict[str, Any] = {}
_FK_JOIN_CACHE: Dict[str, Any] = {}
_SCHEMA_INDEX_CACHE: Optional[Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]] = None
_SCHEMA_INDEX_SOURCE: Optional[str] = None


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    script_candidate = SCRIPT_DIR / candidate
    if script_candidate.exists():
        return script_candidate
    return Path.cwd() / candidate

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
    resolved_path = str(_resolve_path(path))
    if _DATA_DICTIONARY_CACHE.get("path") == resolved_path:
        return _DATA_DICTIONARY_CACHE.get("data", {})

    path_obj = Path(resolved_path)
    if not path_obj.exists():
        logger.warning("Data dictionary file not found: %s", path_obj)
        _DATA_DICTIONARY_CACHE = {"path": resolved_path, "data": {}}
        return {}

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
            _DATA_DICTIONARY_CACHE = {"path": resolved_path, "data": data}
            logger.info("Loaded data dictionary: %s tables from %s", len(data) if isinstance(data, dict) else 0, path_obj)
            return data
    except Exception:
        logger.exception("Failed to load data dictionary: %s", path_obj)
        _DATA_DICTIONARY_CACHE = {"path": resolved_path, "data": {}}
        return {}


def load_json_file(path: str) -> Any:
    """Load any JSON file and cache by path."""
    resolved_path = str(_resolve_path(path))
    cache = _QUERY_EXAMPLES_CACHE if path == QUERY_EXAMPLES_PATH else _FK_JOIN_CACHE if path == FK_JOIN_PATH else None
    if cache is not None and cache.get("path") == resolved_path:
        return cache.get("data")

    path_obj = Path(resolved_path)
    if not path_obj.exists():
        if cache is not None:
            cache["path"] = resolved_path
            cache["data"] = None
        logger.warning("JSON file not found: %s", path_obj)
        return None

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if cache is not None:
                cache["path"] = resolved_path
                cache["data"] = data
            logger.info("Loaded JSON file: %s", path_obj)
            return data
    except Exception:
        if cache is not None:
            cache["path"] = resolved_path
            cache["data"] = None
        logger.exception("Failed to load JSON file: %s", path_obj)
        return None


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


def _get_schema_index(data_dict: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    global _SCHEMA_INDEX_CACHE, _SCHEMA_INDEX_SOURCE
    source_fingerprint = f"{len(data_dict)}:{','.join(sorted(data_dict.keys())[:10])}"
    if _SCHEMA_INDEX_CACHE is not None and _SCHEMA_INDEX_SOURCE == source_fingerprint:
        return _SCHEMA_INDEX_CACHE

    _SCHEMA_INDEX_CACHE = _build_schema_index(data_dict)
    _SCHEMA_INDEX_SOURCE = source_fingerprint
    logger.info("Built schema index for %s tables", len(data_dict))
    return _SCHEMA_INDEX_CACHE


def _select_candidate_tables(
    question: str,
    data_dict: Dict[str, Any],
    max_tables: int = 5,
) -> List[str]:
    """Pick relevant tables based on question keywords, descriptions, and sample values."""
    if not data_dict:
        return []

    question_tokens = set(_tokenize(question))
    table_index, column_index = _get_schema_index(data_dict)

    scores: Dict[str, int] = {}
    for tok in question_tokens:
        # Table name or description matches are the strongest signal
        for tbl in table_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 6
        # Column names, descriptions, and sample values are also important
        for tbl in column_index.get(tok, []):
            scores[tbl] = scores.get(tbl, 0) + 4

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if ranked:
        selected = [t for t, score in ranked if score > 0][:max_tables]
        logger.info("Candidate tables selected via keyword scoring: %s", selected)
        return selected

    fallback_scores: Dict[str, int] = {}
    normalized_question = question.lower()
    for table_name, info in data_dict.items():
        table_name_lc = table_name.lower()
        if any(token in table_name_lc for token in question_tokens) or any(token in normalized_question for token in _tokenize(table_name_lc)):
            fallback_scores[table_name] = fallback_scores.get(table_name, 0) + 3

        for col in info.get("columns", []):
            col_name = str(col.get("name", "")).lower()
            col_desc = str(col.get("col_description", "")).lower()
            if any(token in col_name for token in question_tokens) or any(token in normalized_question for token in _tokenize(col_name)):
                fallback_scores[table_name] = fallback_scores.get(table_name, 0) + 2
            if col_desc and any(token in col_desc for token in question_tokens):
                fallback_scores[table_name] = fallback_scores.get(table_name, 0) + 1

    fallback_ranked = sorted(fallback_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = [t for t, score in fallback_ranked if score > 0][:max_tables]
    if selected:
        logger.info("Candidate tables selected via fallback scoring: %s", selected)
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
            sample_values = [str(x) for x in (col.get("_prompt_samples") or []) if x is not None]
            if col_desc:
                blocks.append(f"    - {name} ({col_type}): {col_desc}")
            else:
                blocks.append(f"    - {name} ({col_type})")
            if sample_values:
                samples = ", ".join(sample_values[:3])
                blocks.append(f"      sample_values: {samples}")

    return "\n".join(blocks)


def _clean_query(query: str) -> str:
    """Clean up SQL query: remove extra whitespace and normalize newlines."""
    # Replace multiple newlines with single space, preserve structure
    query = re.sub(r'\s+', ' ', query.strip())
    return query


def _load_query_examples(path: str) -> Optional[List[Dict[str, Any]]]:
    examples = load_json_file(path)
    if examples is None:
        return None
    if isinstance(examples, list):
        return [e for e in examples if isinstance(e, dict)]
    return []


def _select_query_examples(question: str, examples: List[Dict[str, Any]], max_examples: int = 3) -> List[Dict[str, Any]]:
    question_tokens = set(_tokenize(question))
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for example in examples:
        example_question = str(example.get("question", ""))
        example_tokens = set(_tokenize(example_question))
        if not example_tokens:
            continue
        overlap = len(question_tokens & example_tokens)
        score = overlap
        if example_question.lower() in question.lower() or question.lower() in example_question.lower():
            score += 5
        if example_tokens.issubset(question_tokens) or question_tokens.issubset(example_tokens):
            score += 3
        if score > 0:
            scored.append((score, example))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("question", ""))))
    return [example for _, example in scored[:max_examples]]


def _find_direct_query_example(question: str, examples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a direct matching sample query when the user's question is similar enough."""
    if not examples:
        return None

    normalized_question = question.strip().lower()
    question_tokens = set(_tokenize(normalized_question))
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0

    for example in examples:
        example_question = str(example.get("question", "")).strip()
        if not example_question:
            continue

        normalized_example = example_question.lower()
        example_tokens = set(_tokenize(normalized_example))
        if not example_tokens:
            continue

        if normalized_example == normalized_question or normalized_example in normalized_question or normalized_question in normalized_example:
            return example

        overlap = len(question_tokens & example_tokens)
        threshold = max(3, len(example_tokens) // 2, len(question_tokens) // 2)
        if overlap >= threshold and overlap > best_score:
            best_score = overlap
            best_match = example

    return best_match


def _format_query_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""
    blocks: List[str] = [
        "Example query patterns:",
        "Use these examples as design patterns. If the current question matches one, follow the same SQL style and filtering logic.",
    ]
    for idx, example in enumerate(examples, start=1):
        blocks.append(f"Example {idx}:")
        blocks.append(f"  question: {example.get('question', '').strip()}")
        query_text = example.get('query', '').strip()
        if query_text:
            blocks.append(f"  query: {query_text}")
        reasoning = example.get('reasoning', '').strip()
        if reasoning:
            blocks.append(f"  reasoning: {reasoning}")
    return "\n".join(blocks)


def _load_fk_join_map(path: str) -> Optional[Dict[str, List[str]]]:
    join_map = load_json_file(path)
    if join_map is None:
        return None
    if isinstance(join_map, dict):
        return {str(k): list(v) for k, v in join_map.items() if isinstance(v, list)}
    return {}


def _format_fk_join_hints(tables: List[str], join_map: Dict[str, List[str]]) -> str:
    hints: List[str] = []
    seen: Set[str] = set()
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            t1, t2 = tables[i], tables[j]
            key = f"{t1}|{t2}"
            alt_key = f"{t2}|{t1}"
            columns = join_map.get(key) or join_map.get(alt_key)
            if columns:
                join_key = tuple(sorted([t1, t2]))
                if join_key in seen:
                    continue
                seen.add(join_key)
                hint_columns = ", ".join(columns)
                hints.append(f"- {t1} JOIN {t2} on columns: {hint_columns}")
    if not hints:
        return ""
    return "Join hints for selected tables:\n" + "\n".join(hints)


def _extract_recent_texts(messages: List[BaseMessage], max_texts: int = 2) -> List[str]:
    texts: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        content = get_message_text(msg)
        if content:
            texts.append(content.strip())
        if len(texts) >= max_texts:
            break
    return list(reversed(texts))


def _is_follow_up_question(text: str) -> bool:
    follow_up_words = {
        "it",
        "that",
        "those",
        "these",
        "same",
        "also",
        "then",
        "again",
        "previous",
        "before",
        "continue",
        "follow",
        "still",
        "and",
        "or",
    }
    tokens = set(_tokenize(text))
    return len(tokens & follow_up_words) >= 1


def _apply_result_limit(query: str, top_k: int) -> str:
    """Ensure query returns at most top_k rows (only add if not present)."""
    query = _clean_query(query)
    # Remove trailing semicolon for SQLAlchemy/database compatibility
    query = query.rstrip(";")
    lc = query.lower()
    # Don't add limit if one already exists
    if "limit" in lc or "fetch first" in lc or "rownum" in lc or "offset" in lc:
        return query
    # Oracle-compatible default row limiting
    return query + f" FETCH FIRST {top_k} ROWS ONLY"


# ============================================================================
# System Prompts
# ============================================================================

GENERATE_QUERY_SYSTEM_PROMPT = """
You are an expert SQL generator for a large Oracle-backed reporting system.
Given an input question and a small schema slice, output exactly one safe SQL query.

Rules:
1. Use only the tables and columns shown in the schema context.
2. Prefer the minimum columns and joins needed to answer the question.
3. Use Oracle-safe row limiting: FETCH FIRST {top_k} ROWS ONLY unless the user explicitly asks for all rows.
4. Do not invent table names, column names, filters, or joins.
5. Do not output explanations, markdown, code fences, or comments.
6. Do not output any DML/DDL statements.
7. If the schema context is insufficient, return exactly: ERROR: insufficient schema context.
8. If a similar example query is present, follow its structure but adapt only to the provided schema.
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

def generate_sql_query(state: MessagesState) -> Dict[str, Any]:
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
    query_examples = _load_query_examples(QUERY_EXAMPLES_PATH)
    if not query_examples:
        error_message = (
            "Required configuration is missing. The query examples file could not be loaded or is empty. "
            "Please ensure QUERY_EXAMPLES_PATH is set and the file exists."
        )
        return {"messages": messages + [AIMessage(content=error_message)]}

    logger.info("Incoming question: %s", user_question)
    logger.info("Loaded data dictionary tables: %s", len(data_dict))
    logger.info("Loaded query examples: %s", len(query_examples))

    # *** CHECK FOR DIRECT SAMPLE QUERY MATCH FIRST ***
    # If user's question exactly matches a sample query, use it immediately without schema validation
    matched_example = _find_direct_query_example(user_question, query_examples)
    sample_query = None
    if matched_example:
        sample_query = str(matched_example.get("query", "")).strip()
        sample_query = _clean_query(sample_query)
        if not _is_sql_query(sample_query):
            matched_example = None
            sample_query = None
        else:
            sample_query = _apply_result_limit(sample_query, TOP_K_DEFAULT)
            logger.info(
                "Direct example match found for question='%s' using example question='%s'",
                user_question,
                matched_example.get("question", ""),
            )

    # If we have a direct match, use it immediately before schema checks
    if matched_example and sample_query:
        reasoning_message = ToolMessage(
            content=(
                f"Direct match found in sample queries.\n"
                f"- Question: {user_question}\n"
                f"- Source: {QUERY_EXAMPLES_PATH}"
            ),
            tool_call_id="direct_sample_match"
        )
        tool_call = {
            "name": "sql_db_query",
            "args": {"query": sample_query},
            "id": "query_call",
            "type": "tool_call",
        }
        return {
            "messages": messages + [
                reasoning_message,
                AIMessage(
                    content=(
                        "Found a matching query pattern in the sample queries file. "
                        f"Using this query:\n```sql\n{sample_query}\n```"
                    ),
                    tool_calls=[tool_call]
                )
            ]
        }

    # *** IF NO DIRECT MATCH, PROCEED WITH SCHEMA-BASED QUERY GENERATION ***
    candidate_tables = _select_candidate_tables(user_question, data_dict, max_tables=8)
    logger.info("Selected candidate tables: %s", candidate_tables)

    if not candidate_tables:
        clarification = (
            "I need a little more detail to answer this safely. Please clarify the metric, table subject, or date range."
        )
        return {"messages": messages + [AIMessage(content=clarification)]}

    schema_hint = _format_schema_for_prompt(data_dict, candidate_tables)
    logger.info("Schema hint length: %s characters", len(schema_hint))

    selected_examples = _select_query_examples(user_question, query_examples, max_examples=3)
    examples_hint = _format_query_examples_for_prompt(selected_examples)
    if selected_examples:
        logger.info(
            "Selected example questions: %s",
            [example.get("question", "") for example in selected_examples],
        )

    system_prompt = GENERATE_QUERY_SYSTEM_PROMPT.format(dialect=dialect, top_k=TOP_K_DEFAULT)
    system_prompt += (
        "\n\nFollow this workflow: infer the business intent from the user question, choose only the most relevant tables, "
        "and write the query with Oracle-safe syntax. If you cannot answer from the supplied schema, return exactly: ERROR: insufficient schema context."
    )
    if examples_hint:
        system_prompt += "\n\n" + examples_hint
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

    recent_texts = _extract_recent_texts(messages, max_texts=2)
    user_prompt_messages = []
    if recent_texts:
        if len(recent_texts) > 1 and _is_follow_up_question(recent_texts[-1]):
            user_prompt_messages = [
                {"role": "user", "content": recent_texts[-2]},
                {"role": "user", "content": recent_texts[-1]},
            ]
        else:
            user_prompt_messages = [{"role": "user", "content": recent_texts[-1]}]

    system_message = {"role": "system", "content": system_prompt}
    response = llm.invoke([system_message] + user_prompt_messages)
    query = getattr(response, "content", str(response)).strip()
    logger.info("Raw model output: %s", query[:1000])

    if query.startswith("```sql"):
        query = query[6:]
    if query.startswith("```"):
        query = query[3:]
    if query.endswith("```"):
        query = query[:-3]

    query = _clean_query(query)
    if query.strip().lower().startswith("error:"):
        logger.info("Model returned a schema-insufficient error for question: %s", user_question)
        return {
            "messages": messages + [
                reasoning_message,
                AIMessage(content=query)
            ]
        }
    if not _is_sql_query(query):
        logger.warning("Model output was not valid SQL for question='%s': %s", user_question, query)
        return {
            "messages": messages + [
                reasoning_message,
                AIMessage(content="ERROR: the model did not generate a valid SQL query.")
            ]
        }

    query = _apply_result_limit(query, TOP_K_DEFAULT)
    logger.info("Final query sent to execution: %s", query)

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


def check_query_safety(state: MessagesState) -> Dict[str, Any]:
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
    logger.info("Safety check raw output: %s", checked_query[:1000])
    
    # Clean up query: remove markdown code blocks and semicolons
    if checked_query.startswith("```sql"):
        checked_query = checked_query[6:]
    if checked_query.startswith("```"):
        checked_query = checked_query[3:]
    if checked_query.endswith("```"):
        checked_query = checked_query[:-3]
    
    # Remove semicolons and normalize whitespace
    checked_query = _clean_query(checked_query).rstrip(";")
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


def execute_query(state: MessagesState) -> Dict[str, Any]:
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
    logger.info("Executing query: %s", query)

    # Execute query using the toolkit tool
    try:
        result = run_query_tool.invoke(query)
        result_str = str(result).strip()
        logger.info("Execution result preview: %s", result_str[:1000])

        # Detect errors in execution output
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

        # Detect zero-result responses
        no_rows = False
        if isinstance(result, list):
            no_rows = len(result) == 0
        elif isinstance(result, dict):
            if result.get("rows") == [] or result.get("data") == []:
                no_rows = True
        elif hasattr(result, "rowcount"):
            try:
                no_rows = getattr(result, "rowcount") == 0
            except Exception:
                no_rows = False
        elif hasattr(result, "rows"):
            try:
                no_rows = len(getattr(result, "rows")) == 0
            except Exception:
                no_rows = False

        if not is_error and not no_rows:
            if result_str.lower() in ("", "[]") or "no rows" in result_str.lower() or "0 rows" in result_str.lower():
                no_rows = True

        if is_error:
            result_message = ToolMessage(
                content=f"Error executing query:\n\n{result_str}",
                tool_call_id=tool_call.get("id", "query_call")
            )
        elif no_rows:
            result_message = ToolMessage(
                content=(
                    "Query executed successfully, but returned 0 rows. "
                    "This means the SQL is syntactically valid, but the current data does not contain matching records. "
                    "Please review the query filters or sample availability and consider revising the question or query.\n\n"
                    f"Query:\n```sql\n{query}\n```\n\n"
                    f"Output:\n{result_str}"
                ),
                tool_call_id=tool_call.get("id", "query_call")
            )
        else:
            result_message = ToolMessage(
                content=f"Query executed successfully.\n\nResults:\n{result_str}",
                tool_call_id=tool_call.get("id", "query_call")
            )
    except Exception as e:
        logger.exception("Query execution failed")
        result_message = ToolMessage(
            content=f"Error executing query: {str(e)}",
            tool_call_id=tool_call.get("id", "query_call")
        )

    return {"messages": messages + [result_message]}


# ============================================================================
# Router Functions
# ============================================================================

def should_check_query(state: MessagesState) -> Literal["check_query", "execute_query"]:
    """Route to check_query step."""
    return "check_query"


def should_execute(state: MessagesState) -> Literal["execute_query", END]:
    """Always execute after checking."""
    return "execute_query"


# ============================================================================
# Build LangGraph StateGraph
# ============================================================================

def build_agent() -> StateGraph:
    """Build and return the compiled LangGraph agent."""
    
    # Create the state graph
    builder = StateGraph(MessagesState)

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
    logger.info("Agent startup: data_dictionary=%s query_examples=%s fk_join=%s", DATA_DICTIONARY_PATH, QUERY_EXAMPLES_PATH, FK_JOIN_PATH)

    data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)
    if data_dict:
        tables = sorted(data_dict.keys())
        print(f"Loaded schema for {len(tables)} tables from {DATA_DICTIONARY_PATH}")
        print("Example tables:", ", ".join(tables[:10]))
        logger.info("Startup loaded %s tables", len(tables))
    else:
        print("No data dictionary found.")
        logger.warning("Startup did not load any tables from the data dictionary")

    question = input("\nEnter a natural language question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    # Invoke agent with the question
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # Print the conversation
    print("\n" + "=" * 60)
    for msg in result.get("messages", []):
        role = "Agent" if isinstance(msg, AIMessage) else "System"
        print(f"\n[{role}]:\n{msg.content[:500]}")


if __name__ == "__main__":
    main()

