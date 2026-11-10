"""
FastAPI Production Microservice for Agentic-DeepResearch.

Provides RESTful endpoints for autonomous multi-agent research synthesis,
health telemetry, and interactive web dashboard hosting.
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from graph.workflow import build_graph, create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgenticDeepResearch.API")

app = FastAPI(
    title="Agentic-DeepResearch API",
    description="Autonomous Multi-Agent Web Research & Report Engine with LangGraph, Gemini & Self-Correction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external frontend or microservice integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph workflow once during startup
research_graph = build_graph()


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The research question to investigate")


class ResearchResponse(BaseModel):
    question: str
    language: str
    answer: str
    sub_questions: List[str]
    sources: List[str]
    revisions: int
    is_satisfactory: bool
    status: str = "completed"


class HealthResponse(BaseModel):
    status: str
    version: str
    author: str
    github: str
    providers: dict
    active_model: str


@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """
    Submits a research inquiry through the multi-agent cyclic workflow:
    Planner -> Searcher -> Writer -> Critic (loop) -> Synthesis.
    """
    initial_state = create_initial_state(request.question)

    try:
        logger.info(f"Initiating research inquiry: '{request.question[:60]}...'")
        result = await research_graph.ainvoke(initial_state)

        return ResearchResponse(
            question=request.question,
            language=result.get("language", "en"),
            answer=result.get("final_answer", ""),
            sub_questions=result.get("sub_questions", []),
            sources=result.get("sources", []),
            revisions=result.get("revision_count", 0),
            is_satisfactory=result.get("is_satisfactory", True),
            status="completed"
        )
    except Exception as e:
        logger.error(f"Research execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns microservice health telemetry and available LLM/Search providers."""
    google_key_set = bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_API_KEY") != "your_gemini_api_key_here")
    openai_key_set = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
    tavily_key_set = bool(os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "your_tavily_api_key_here")

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        author="Mohammed AL-Haj",
        github="https://github.com/alhaj-ai/Agentic-DeepResearch",
        providers={
            "gemini": google_key_set,
            "openai": openai_key_set,
            "tavily": tavily_key_set,
            "wikipedia_fallback": True
        },
        active_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )


@app.get("/agents/graph")
def graph_topology():
    """Returns the architectural topology of the multi-agent graph."""
    return {
        "architecture": "LangGraph Directed Cyclic Graph",
        "nodes": [
            {"id": "planner", "role": "Query deconstruction into 2-3 focused search vectors"},
            {"id": "searcher", "role": "Parallel factual retrieval across Tavily and Open Web APIs"},
            {"id": "writer", "role": "Evidence-grounded synthesis with executive formatting"},
            {"id": "critic", "role": "Rigid multi-dimension fact-checking and quality verdict"},
            {"id": "finalize", "role": "Report formatting and citation delivery"}
        ],
        "cyclic_condition": "Critic checks quality; loops to Writer up to 3 revisions if deficient."
    }


# Mount the frontend interface
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
