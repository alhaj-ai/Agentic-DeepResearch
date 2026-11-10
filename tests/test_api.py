"""
REST API endpoint tests for Agentic-DeepResearch.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["author"] == "Mohammed AL-Haj"
    assert "github.com/alhaj-ai" in data["github"]


def test_agents_graph_endpoint():
    response = client.get("/agents/graph")
    assert response.status_code == 200
    data = response.json()
    assert "LangGraph" in data["architecture"]
    assert len(data["nodes"]) >= 4


def test_research_endpoint():
    payload = {"question": "What is reinforcement learning from human feedback (RLHF)?"}
    response = client.post("/research", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert len(data["answer"]) > 0
    assert len(data["sub_questions"]) >= 1
