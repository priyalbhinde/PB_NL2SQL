import json
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI

DB_URL = ""  # Fill in your database URL
LLM_CONFIG = {
    "model_name": "gpt-oss-120b",
    "api_key": "",  # Fill in your API key
    "base_url": "",  # Fill in your base URL
    "proxy_url": "http://ncproxy1:8080"
}
TEST_TABLES = [
  
   
]
DOMAIN_CONTEXT = ""
SAMPLE_VALUES_PER_COLUMN = 5
OUTPUT_FILE = "data_dictionary.json"

def load_llm() -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "model_name": LLM_CONFIG["model_name"],
        "api_key": LLM_CONFIG["api_key"],
        "base_url": LLM_CONFIG["base_url"],
        "temperature": LLM_CONFIG.get("temperature", 0),
    }
    proxy = LLM_CONFIG.get("proxy_url")
    if proxy:
        warnings.filterwarnings("ignore", category=UserWarning, module="httpx")
        kwargs["http_client"] = httpx.Client(proxy=proxy, verify=False)
    return ChatOpenAI(**kwargs)

def table_has_at_least_n_rows(engine: Engine, table_name: str, n: int = 5) -> bool:
    """
    Returns True if the optimizer can find n rows.
    The query stops as soon as n rows are produced, so it never scans the whole table.
    """
    sql = f'SELECT 1 FROM {table_name} WHERE ROWNUM <= {n}'
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return len(rows) >= 1
    except Exception as exc:
        print(f"[WARN] existence check failed for {table_name}: {exc}")
        return False

def extract_table_schema(engine: Engine, table_name: str) -> Dict[str, Any]:
    """
    Pull column metadata, PK info and (if the table has rows) a few distinct, non-NULL sample values (max SAMPLE_VALUES_PER_COLUMN per column).
    """
    print(f"\n=> Extracting schema for: {table_name}")
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise ValueError(f"Table {table_name} not found in the current schema")
    columns = inspector.get_columns(table_name)
    pk_info = inspector.get_pk_constraint(table_name)
    pk_cols = pk_info.get("constrained_columns") or []
    has_rows = table_has_at_least_n_rows(engine, table_name, n=5)
    enriched_columns: List[Dict[str, Any]] = []
    with engine.connect() as conn:
        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            sample_vals: List[str] = []
            if has_rows:
                try:
                    sample_sql = text(
                        f'SELECT DISTINCT "{col_name}" '
                        f'FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL '
                        f'FETCH FIRST {SAMPLE_VALUES_PER_COLUMN} ROWS ONLY'
                    )
                    rows = conn.execute(sample_sql).fetchall()
                    sample_vals = [str(r[0]) for r in rows]
                except Exception as exc:
                    sample_vals = [f"(error: {exc})"]
            enriched_columns.append({
                "name": col_name,
                "type": col_type,
                "nullable": col.get("nullable", True),
                "is_pk": col_name in pk_cols,
                "_prompt_samples": sample_vals,
            })
    return {
        "table_name": table_name,
        "columns": enriched_columns,
        "has_rows": has_rows,
    }

def enrich_table_with_llm(raw_schema: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    table_name = raw_schema["table_name"]
    print(f" -> LLM enriching: {table_name}")
    col_lines: List[str] = []
    for col in raw_schema["columns"]:
        pk_tag = " [PK]" if col["is_pk"] else ""
        samples = ", ".join(col.get("_prompt_samples", [])[:3]) or "no samples"
        col_lines.append(
            f' - {col["name"]} ({col["type"]}){pk_tag} | samples: {samples}'
        )
    col_block = "\n".join(col_lines)
    prompt = f"You are a senior {DOMAIN_CONTEXT} data analyst and database documentation expert.\n"
    prompt += f"A company has a {DOMAIN_CONTEXT} database table described below. Your job is to generate clear, business friendly documentation.\n"
    prompt += f"Table name: {table_name}\n"
    prompt += f"Has rows?: {'YES' if raw_schema.get('has_rows') else 'NO'}\nColumns:\n{col_block}\n"
    prompt += "Generate the documentation in this exact JSON format (no markdown, no extra text):\n"
    prompt += '{\n"table_description": "2-3 sentence business description of what this table stores and its purpose in a {DOMAIN_CONTEXT} context",\n"column_descriptions": {\n"column_name": "concise business description, max 15 words"\n}\n}'
    start = time.time()
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        latency = round(time.time() - start, 2)
        content = response.content.strip()
        if content.startswith("```json"):
            content = "\n".join(content.split("\n")[1:-1])
        descriptions = json.loads(content)
        enriched_columns = []
        for col in raw_schema["columns"]:
            enriched_columns.append({
                "name": col["name"],
                "type": col["type"],
                "nullable": col["nullable"],
                "is_pk": col["is_pk"],
                "col_description": descriptions.get("column_descriptions", {}).get(col["name"], "")
            })
        return {
            **raw_schema,
            "table_description": descriptions.get("table_description", ""),
            "columns": enriched_columns,
            "enrichment_metadata": {
                "llm_model": LLM_CONFIG["model_name"],
                "enriched_at": datetime.utcnow().isoformat(),
                "llm_latency_seconds": latency,
                "status": "success",
            }
        }
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed for {table_name}: {e}")
        print(f"Raw LLM response (first 300 chars): {response.content[:300]}...")
        return {
            **raw_schema,
            "table_description": "",
            "enrichment_metadata": {
                "status": "failed_json_parse",
                "error": str(e),
                "raw_response_preview": response.content[:300],
            }
        }
    except Exception as e:
        print(f"[ERROR] LLM call failed for {table_name}: {e}")
        return {
            **raw_schema,
            "table_description": "",
            "enrichment_metadata": {"status": "failed", "error": str(e)},
        }

def print_report(dictionary: Dict[str, Any], elapsed_total: float) -> None:
    for table_name, info in dictionary.items():
        meta = info.get("enrichment_metadata", {})
        status = meta.get("status", "unknown")
        flag = "YES" if status == "success" else "NO"
        print(f"\n{flag} TABLE: {table_name}")
        print(f" Description: {info.get('table_description', 'N/A')}")
        print(f" Has rows?: {'YES' if info.get('has_rows') else 'NO'}")
        print(f" Columns: {len(info.get('columns', []))}")
        print(f" LLM Latency: {meta.get('llm_latency_seconds', 'N/A')} s")
        print("Column Descriptions (first 8):")
        for col in info.get("columns", [])[:8]:
            desc = col.get("col_description", "(none)")
            pk = " [PK]" if col.get("is_pk") else ""
            print(f" {col['name']}{pk}: {desc}")
        remaining = len(info.get("columns", [])) - 8
        if remaining > 0:
            print(f".. and {remaining} more columns (see JSON output)")
    print(f"\nTotal runtime: {elapsed_total:.1f}s")

def main() -> None:
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    llm = load_llm()
    dictionary: Dict[str, Any] = {}
    start_total = time.time()
    for table_name in TEST_TABLES:
        print(f"\nProcessing {table_name}")
        try:
            raw_schema = extract_table_schema(engine, table_name)
            enriched = enrich_table_with_llm(raw_schema, llm)
            dictionary[table_name] = enriched
        except Exception as e:
            print(f"[ERROR] Failed entirely for {table_name}: {e}")
            dictionary[table_name] = {
                "table_name": table_name,
                "error": str(e),
                "enrichment_metadata": {"status": "failed", "error": str(e)},
            }
        time.sleep(1)
    elapsed = round(time.time() - start_total, 1)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {OUTPUT_FILE}")
    print_report(dictionary, elapsed)

if __name__ == "__main__":
    main()
