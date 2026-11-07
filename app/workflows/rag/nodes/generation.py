"""LangGraph Generation node backed by GenerationAgent."""

from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

from app.agents.Generation import GenerationAgent
from app.workflows.rag.state import RAGWorkflowState


def create_generation_node(
    agent: GenerationAgent,
) -> Callable[[RAGWorkflowState], Any]:
    """Create a generation node; optionally publish provisional SSE deltas."""

    async def generation_node(state: RAGWorkflowState) -> dict[str, Any]:
        if state.get("streaming"):
            writer = get_stream_writer()
            final_output = None
            async for event in agent.astream_generate(
                current_query=state["query"],
                retrieved_documents=state.get("retrieved_documents") or [],
                conversation_history=state.get("context"),
            ):
                if event.event_type.value == "content_delta" and event.delta:
                    writer({"type": "delta", "delta": event.delta, "segment_index": event.segment_index})
                elif event.event_type.value == "completed":
                    final_output = event.output
            if final_output is None:
                raise RuntimeError("Generation stream ended without a validated output.")
            return {"generation_output": final_output}

        output = await agent.agenerate(
            current_query=state["query"],
            retrieved_documents=state.get("retrieved_documents") or [],
            conversation_history=state.get("context"),
        )
        return {"generation_output": output}

    return generation_node
