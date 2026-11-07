from typing import Any, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph

from app.llm.multi_query_client import multi_query_llm_client

from .config import MultiQueryGeneratorConfig
from .prompts import get_multi_query_generator_prompt
from .schemas import (
    AvailableDocuments,
    GeneratedQuery,
    MultiQueryGeneratorOutput,
    QueryVersionScope,
)
from .state import MultiQueryGeneratorState


class MultiQueryGeneratorError(Exception):
    """Base exception for multi-query generation errors."""


class EmptyQueryError(MultiQueryGeneratorError):
    """Raised when the current query is empty."""


class MultiQueryGeneratorAgent:

    def __init__(
        self,
        config: MultiQueryGeneratorConfig | None = None,
        client=None,
        model=None,
    ):
        self.config = config or MultiQueryGeneratorConfig()

        self.client = client
        self.model = model

        if self.model is None:
            if self.client is None:
                self.model = multi_query_llm_client.get_llm()
            elif hasattr(self.client, "get_llm"):
                self.model = self.client.get_llm()
            else:
                self.model = self.client

        self.model = self.model.bind(
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )

        self.parser = PydanticOutputParser(
            pydantic_object=MultiQueryGeneratorOutput
        )

        self.prompt = get_multi_query_generator_prompt(
            max_queries=self.config.max_queries
        )

        self.chain = self.prompt | self.model | self.parser

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MultiQueryGeneratorState)

        workflow.add_node(
            "generate_queries",
            self._generate_queries,
        )

        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", END)

        return workflow.compile()

    def _generate_queries(
        self,
        state: MultiQueryGeneratorState,
    ) -> dict:

        current_query = state["current_query"].strip()

        if not current_query:
            raise EmptyQueryError("Current query cannot be empty.")

        conversation_history = state.get("conversation_history", [])
        available_documents = AvailableDocuments.model_validate(
            state["available_documents"]
        )

        result = self.chain.invoke(
            {
                "current_query": current_query,
                "conversation_history": conversation_history,
                "available_documents": available_documents.model_dump(),
                "format_instructions": self.parser.get_format_instructions(),
            },
            config={
                "metadata": {
                    "agent": "multi_query_generator",
                }
            },
        )

        return {
            "generated_queries": self._validate_generated_queries(
                result,
                available_documents,
            ),
        }

    def _validate_generated_queries(
        self,
        result: MultiQueryGeneratorOutput,
        available_documents: AvailableDocuments,
    ) -> list[GeneratedQuery]:
        available_by_status = {
            "active": {
                document.version_year
                for document in available_documents.active_documents
            },
            "superseded": {
                document.version_year
                for document in available_documents.superseded_documents
            },
        }
        available_years = (
            available_by_status["active"]
            | available_by_status["superseded"]
        )
        validated = []
        for generated_query in result.queries[:self.config.max_queries]:
            target_years = set(generated_query.target_versions)
            if generated_query.version_scope == QueryVersionScope.ACTIVE:
                allowed_years = available_by_status["active"]
            elif generated_query.version_scope == QueryVersionScope.SUPERSEDED:
                allowed_years = available_by_status["superseded"]
            else:
                allowed_years = available_years

            if not target_years.issubset(allowed_years):
                if target_years.issubset(available_years):
                    raise MultiQueryGeneratorError(
                        "Generated query target years do not match its "
                        "version scope."
                    )
                raise MultiQueryGeneratorError(
                    "Generated query targets unavailable document versions."
                )
            validated.append(generated_query)
        return validated

    def generate(
        self,
        current_query: str,
        conversation_history: Sequence[BaseMessage] | None = None,
        available_documents: AvailableDocuments | dict[str, Any] | None = None,
    ) -> list[GeneratedQuery]:

        if not current_query or not current_query.strip():
            raise EmptyQueryError("Current query cannot be empty.")

        history = list(conversation_history or [])
        catalogue = AvailableDocuments.model_validate(
            available_documents or {}
        )

        result = self.graph.invoke(
            {
                "current_query": current_query,
                "conversation_history": history,
                "available_documents": catalogue,
                "generated_queries": [],
            }
        )

        return result["generated_queries"]

    async def agenerate(
        self,
        current_query: str,
        conversation_history: Sequence[BaseMessage] | None = None,
        available_documents: AvailableDocuments | dict[str, Any] | None = None,
    ) -> list[GeneratedQuery]:

        if not current_query or not current_query.strip():
            raise EmptyQueryError("Current query cannot be empty.")

        history = list(conversation_history or [])
        catalogue = AvailableDocuments.model_validate(
            available_documents or {}
        )

        result = await self.graph.ainvoke(
            {
                "current_query": current_query,
                "conversation_history": history,
                "available_documents": catalogue,
                "generated_queries": [],
            },
        )

        return result["generated_queries"]