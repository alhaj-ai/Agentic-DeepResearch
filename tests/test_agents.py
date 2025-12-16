"""
Unit tests for individual agents in Agentic-DeepResearch.
"""

import pytest
from agents.llm_client import detect_language
from agents.planner import planner_agent
from agents.searcher import searcher_agent
from agents.writer import writer_agent
from agents.critic import critic_agent
from graph.workflow import create_initial_state


def test_language_detection():
    arabic_text = "ما هي أحدث بنى النظم متعددة الوكلاء والذكاء الاصطناعي؟"
    english_text = "What are the latest architectures for multi-agent LLM systems?"
    mixed_text = "كيف أستخدم LangGraph مع Python؟"

    assert detect_language(arabic_text) == "ar"
    assert detect_language(english_text) == "en"
    assert detect_language(mixed_text) == "ar"


def test_planner_agent():
    state = create_initial_state("How do multi-agent systems handle task decomposition?")
    output = planner_agent(state)

    assert "sub_questions" in output
    assert len(output["sub_questions"]) >= 1
    assert output["language"] == "en"


def test_planner_agent_arabic():
    state = create_initial_state("ما هي مميزات بنية الوكلاء الذاتية مقارنة بالنماذج الأحادية؟")
    output = planner_agent(state)

    assert "sub_questions" in output
    assert len(output["sub_questions"]) >= 1
    assert output["language"] == "ar"


def test_searcher_agent_fallback():
    state = create_initial_state("Artificial Intelligence in Medical Imaging")
    state["sub_questions"] = ["Deep learning algorithms in radiology", "Clinical validation of diagnostic AI"]
    output = searcher_agent(state)

    assert "search_results" in output
    assert len(output["search_results"]) > 0
    assert "sources" in output
    assert len(output["sources"]) > 0


def test_writer_agent():
    state = create_initial_state("What are key benefits of LangGraph?")
    state["search_results"] = ["Source: https://example.com\nLangGraph enables cyclic agent workflows with explicit state management."]
    output = writer_agent(state)

    assert "draft_answer" in output
    assert len(output["draft_answer"]) > 20


def test_critic_agent_evaluation():
    state = create_initial_state("Explain state machines in agent frameworks.")
    state["draft_answer"] = (
        "### 1. Overview\n"
        "State machines allow agents to maintain persistence across iterative steps.\n"
        "### 2. Benefits\n"
        "- Deterministic transitions\n"
        "- Fault tolerance\n\n"
        "### Sources\n"
        "https://example.com/state-machines"
    )
    output = critic_agent(state)

    assert "is_satisfactory" in output
    assert "critique" in output
    assert isinstance(output["is_satisfactory"], bool)
