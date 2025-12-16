"""
Integration tests for the LangGraph workflow in Agentic-DeepResearch.
"""

import pytest
from graph.workflow import build_graph, should_continue, create_initial_state


def test_graph_compilation():
    graph = build_graph()
    assert graph is not None


def test_should_continue_logic():
    pass_state = create_initial_state("Query")
    pass_state["is_satisfactory"] = True
    assert should_continue(pass_state) == "end"

    fail_state = create_initial_state("Query")
    fail_state["is_satisfactory"] = False
    fail_state["revision_count"] = 1
    assert should_continue(fail_state) == "revise"

    max_revision_state = create_initial_state("Query")
    max_revision_state["is_satisfactory"] = False
    max_revision_state["revision_count"] = 3
    assert should_continue(max_revision_state) == "end"


@pytest.mark.asyncio
async def test_full_workflow_execution():
    graph = build_graph()
    initial_state = create_initial_state("Explain Multi-Agent Orchestration in 2026")

    result = await graph.ainvoke(initial_state)

    assert result is not None
    assert "final_answer" in result
    assert len(result["final_answer"]) > 0
    assert "sub_questions" in result
    assert len(result["sub_questions"]) >= 1
    assert "sources" in result
