"""Utility & helper functions."""

import httpx
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""

    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")

    # Some message types use a list/tuple of segments.
    texts: List[str] = []
    if isinstance(content, (list, tuple)):
        for segment in content:
            if isinstance(segment, str):
                texts.append(segment)
            elif isinstance(segment, dict):
                texts.append(segment.get("text", ""))

    return "".join(texts).strip()

def load_chat_model() -> ChatOpenAI:
    # Configure model
    base_url = "url"
    api_key = "key"
    proxy_url = "proxy"
    http_client = httpx.Client(proxy=proxy_url, verify=False)
    llm = ChatOpenAI(
        # model_name="Meta-Llama-33-70B-Instruct",
        model_name="gpt-oss-120b",
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        temperature=0,
    )
    return llm