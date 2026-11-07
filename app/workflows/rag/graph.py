"""LangGraph construction and compilation for the streamlined RAG workflow."""

from app.agents.Generation import GenerationAgent
from app.services.retrieval.retrieval_service import RetrievalOrchestrator
from app.validation import StaticValidationLayer
from langgraph.graph import END, START, StateGraph

from app.workflows.rag.nodes.generation import create_generation_node
from app.workflows.rag.nodes.input_validation import input_validation_node
from app.workflows.rag.nodes.retriever import create_retriever_node
from app.workflows.rag.nodes.static_validation import create_static_validation_node
from app.workflows.rag.routing import route_after_validation
from app.workflows.rag.state import RAGWorkflowState


def create_rag_graph(
    *,
    retriever: RetrievalOrchestrator,
    generation_agent: GenerationAgent,
    validator: StaticValidationLayer,
):
    """Create a one-pass RAG graph: validate, retrieve, generate, validate output."""
    graph = StateGraph(RAGWorkflowState)
    graph.add_node("input_validation", input_validation_node)
    graph.add_node("retriever", create_retriever_node(retriever))
    graph.add_node("generation", create_generation_node(generation_agent))
    graph.add_node("static_validation", create_static_validation_node(validator))

    graph.add_edge(START, "input_validation")
    graph.add_conditional_edges(
        "input_validation",
        route_after_validation,
        {"valid": "retriever", "invalid": END},
    )
    graph.add_edge("retriever", "generation")
    graph.add_edge("generation", "static_validation")
    graph.add_edge("static_validation", END)
    return graph.compile()
