# Contributing to Agentic-DeepResearch

Thank you for your interest in contributing to **Agentic-DeepResearch**!

## How to Contribute

1. **Fork the Repository**: Click the Fork button on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/Agentic-DeepResearch.git
   cd Agentic-DeepResearch
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install Dependencies & Run Tests**:
   ```bash
   pip install -r requirements.txt
   python -m pytest tests/ -v
   ```
5. **Commit your work**: Follow conventional commits (e.g. `feat: add DuckDuckGo live streaming support`).
6. **Open a Pull Request**: Submit your PR with a clear summary of your changes.

## Areas of Active Development
- Local LLM inference integration (Ollama / vLLM / HuggingFace Transformers).
- Advanced evaluation metrics (RAGAS / DeepEval automated faithfulness benchmarks).
- Streaming UI response via Server-Sent Events (SSE).
