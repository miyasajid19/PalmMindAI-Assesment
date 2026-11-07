from typing import TypedDict

from langchain_core.messages import BaseMessage

from .schemas import AvailableDocuments, GeneratedQuery


class MultiQueryGeneratorState(TypedDict):
    current_query: str
    conversation_history: list[BaseMessage]
    available_documents: AvailableDocuments
    generated_queries: GeneratedQuery