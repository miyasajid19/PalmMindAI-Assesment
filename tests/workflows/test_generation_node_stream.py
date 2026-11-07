
import asyncio

from langgraph.graph import END, START, StateGraph

from app.agents.Generation.schemas import (
    AnswerSegment,
    GenerationOutput,
    GenerationStatus,
    GenerationStreamEvent,
    GenerationStreamEventType,
)
from app.workflows.rag.nodes.generation import create_generation_node
from app.workflows.rag.state import RAGWorkflowState


class FakeStreamingGenerationAgent:
    async def astream_generate(self, **kwargs):
        yield GenerationStreamEvent(
            event_type=GenerationStreamEventType.CONTENT_DELTA,
            segment_index=0,
            delta="hello ",
        )
        yield GenerationStreamEvent(
            event_type=GenerationStreamEventType.CONTENT_DELTA,
            segment_index=0,
            delta="world",
        )
        yield GenerationStreamEvent(
            event_type=GenerationStreamEventType.COMPLETED,
            output=GenerationOutput(
                status=GenerationStatus.ANSWERED,
                segments=[AnswerSegment(text="hello world", citations=[])],
            ),
        )


def test_generation_node_publishes_custom_delta_events_in_graph_stream():
    async def run():
        graph_builder = StateGraph(RAGWorkflowState)
        graph_builder.add_node("generation", create_generation_node(FakeStreamingGenerationAgent()))
        graph_builder.add_edge(START, "generation")
        graph_builder.add_edge("generation", END)
        graph = graph_builder.compile()

        observed_custom = []
        final_state = None
        async for mode, chunk in graph.astream(
            {"query": "hello?", "streaming": True, "retrieved_documents": []},
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                observed_custom.append(chunk)
            elif mode == "values":
                final_state = chunk

        assert observed_custom == [
            {"type": "delta", "delta": "hello ", "segment_index": 0},
            {"type": "delta", "delta": "world", "segment_index": 0},
        ]
        assert final_state["generation_output"].segments[0].text == "hello world"

    asyncio.run(run())
