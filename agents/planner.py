"""
Planner Agent for Agentic-DeepResearch.

Decomposes complex research questions into structured, focused sub-queries.
Supports both Arabic and English native query planning.
"""

from graph.state import AgentState
from agents.llm_client import call_llm, detect_language


def planner_agent(state: AgentState) -> dict:
    """
    Planner Node in LangGraph.
    Breaks down the user's inquiry into 2 focused sub-questions.
    """
    question = state["question"]
    language = detect_language(question)

    if language == "ar":
        system_prompt = """أنت مخطط أبحاث ذكي ومحترف (Research Planner).
مهمتك هي أخذ سؤال المستخدم وتحليله وتفكيكه إلى سؤالين فرعيين مركزين ومحددين فقط (2 Sub-questions).
يجب أن يغطي السؤالان معاً كافة أبعاد السؤال الأصلي بصورة متكاملة باللغة العربية الفصحى.
أرجع الإجابة فقط كقائمة مرقمة بالشكل التالي دون أي مقدمات أو إضافات:
1. السؤال الفرعي الأول
2. السؤال الفرعي الثاني"""
    else:
        system_prompt = """You are a senior research planning agent.
Your job is to analyze the user's research question and break it down into exactly 2 focused, complementary sub-questions.
These sub-questions together must fully cover the core architectural, factual, and practical scope of the original inquiry.
Return ONLY a numbered list in this exact format, with no preamble or outro:
1. First focused sub-question
2. Second focused sub-question"""

    raw_response = call_llm(system_prompt=system_prompt, user_prompt=question, temperature=0.1)

    lines = [line.strip() for line in raw_response.strip().split("\n") if line.strip()]
    sub_questions = []
    for line in lines:
        if line and line[0].isdigit() and ("." in line or "-" in line):
            cleaned = line.split(".", 1)[-1].split("-", 1)[-1].strip()
            if cleaned:
                sub_questions.append(cleaned)
        elif line.startswith("-") or line.startswith("*"):
            cleaned = line[1:].strip()
            if cleaned:
                sub_questions.append(cleaned)

    # Fallback if parsing produced fewer than 2 sub-questions
    if len(sub_questions) < 2:
        if language == "ar":
            sub_questions = [
                f"ما هي المفاهيم والأسس الجوهرية لـ {question}؟",
                f"ما هي التطبيقات العملية وأفضل الممارسات المتعلقة بـ {question}؟"
            ]
        else:
            sub_questions = [
                f"What are the core technical foundations and mechanisms of {question}?",
                f"What are the real-world implementations, benchmarks, and best practices for {question}?"
            ]

    return {
        "sub_questions": sub_questions[:2],
        "language": language
    }
