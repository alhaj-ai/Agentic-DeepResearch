"""
Critic Agent for Agentic-DeepResearch.

Evaluates draft research reports against a strict multi-dimensional rubric:
1. Relevance & Completeness (Does it directly address the query?)
2. Conciseness & Structure (150-350 words, crisp formatting, executive clarity)
3. Factual Grounding & Citations (Backed by retrieved evidence with source URLs)
"""

from graph.state import AgentState
from agents.llm_client import call_llm


def critic_agent(state: AgentState) -> dict:
    """
    Critic Node in LangGraph.
    Inspects draft_answer and decides whether it meets quality standards (PASS)
    or triggers a revision loop back to the Writer (FAIL).
    """
    question = state["question"]
    draft_answer = state.get("draft_answer", "")
    language = state.get("language", "en")
    revision_count = state.get("revision_count", 0)

    if language == "ar":
        system_prompt = """أنت مدقق ومحكم جودة بحثية صارم (Strict Quality & Fact-Checking Reviewer).
قيم التقرير البحثي بناءً على 3 معايير صارمة:
1. الملاءمة والشمول: هل يجيب التقرير مباشرة وبشكل متكامل على السؤال دون استطراد؟
2. الإيجاز والهيكلة: هل التقرير مصاغ بإحكام (بين 150 إلى 350 كلمة)، منظم بأقسام ونقاط سهلة القراءة؟
3. الإسناد والمصادر: هل يحتوي التقرير على روابط مصادر ومراجع حقيقية وموثوقة في نهايته؟

يجب أن تكون إجابتك بالصيغة الدقيقة التالية دون أي تغيير:
VERDICT: PASS أو VERDICT: FAIL
CRITIQUE: شرح تفصيلي للنقاط الإيجابية ونقاط الضعف المطلوب تعديلها."""
    else:
        system_prompt = """You are a rigorous peer-review and fact-checking Critic agent.
Evaluate the draft research answer strictly against 3 criteria:
1. Query Alignment: Does it directly and completely address the original inquiry without drifting?
2. Structure & Length: Is it executive, well-structured with sections/bullets, and within 150-350 words?
3. Source Grounding: Does it end with valid source URLs and maintain factual consistency?

You must format your reply EXACTLY like this:
VERDICT: PASS or FAIL
CRITIQUE: Detailed constructive feedback on what is strong or needs revision."""

    user_prompt = f"""Original Question:
{question}

Draft Answer to Evaluate:
{draft_answer}"""

    raw_response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0
    )

    is_pass = "VERDICT: PASS" in raw_response.upper()

    if is_pass:
        return {
            "is_satisfactory": True,
            "critique": raw_response,
            "final_answer": draft_answer,
            "revision_count": revision_count
        }
    else:
        return {
            "is_satisfactory": False,
            "critique": raw_response,
            "final_answer": draft_answer if revision_count >= 2 else "",
            "revision_count": revision_count + 1
        }
