"""
Workflow Orchestrator for Agentic-DeepResearch.

Builds the cyclic LangGraph state machine:
Planner -> Searcher -> Writer -> Critic -> (loop back to Writer or END).
"""

import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.planner import planner_agent
from agents.searcher import searcher_agent
from agents.writer import writer_agent
from agents.critic import critic_agent

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))


def should_continue(state: AgentState) -> str:
    """
    Evaluator edge routing function.
    Decides whether to finalize the report or trigger a revision loop.
    """
    if state.get("is_satisfactory", False) or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "end"
    return "revise"


def finalize_node(state: AgentState) -> dict:
    """
    Ensures that a non-empty final_answer is always delivered,
    even if the maximum revision threshold was reached.
    """
    final = state.get("final_answer")
    if not final:
        final = state.get("draft_answer", "")
    return {"final_answer": final}


def build_graph():
    """Compiles and returns the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register agent nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("searcher", searcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("finalize", finalize_node)

    # Set workflow entry point
    workflow.set_entry_point("planner")

    # Linear pipelines
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "writer")
    workflow.add_edge("writer", "critic")

    # Dynamic cyclic edge with self-correcting feedback
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "revise": "writer",
            "end": "finalize"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


def create_initial_state(question: str) -> AgentState:
    """Helper to initialize a clean research state."""
    return {
        "question": question,
        "language": "en",
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "draft_answer": "",
        "critique": "",
        "revision_count": 0,
        "is_satisfactory": False,
        "final_answer": ""
    }
