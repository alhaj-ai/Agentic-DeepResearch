"""
Writer Agent for Agentic-DeepResearch.

Synthesizes evidence retrieved by the Searcher into a coherent, executive,
citation-backed research report. Supports iterative revisions guided by Critic feedback.
"""

from graph.state import AgentState
from agents.llm_client import call_llm


def writer_agent(state: AgentState) -> dict:
    """
    Writer Node in LangGraph.
    Synthesizes search results and addresses prior critiques if in a revision loop.
    """
    question = state["question"]
    language = state.get("language", "en")
    search_results = "\n\n".join(state.get("search_results", []))
    critique = state.get("critique", "None")
    revision_count = state.get("revision_count", 0)

    if language == "ar":
        system_prompt = f"""أنت كاتب وباحث علمي محترف ومتخصص في الصياغة التحليلية التنفيذية (Executive Research Writer).
مهمتك هي كتابة تقرير بحثي مركز، عالي الجودة ومستند إلى الأدلة للإجابة عن السؤال.

قواعد الصياغة الأساسية:
1. الإجابة المباشرة والدقيقة على السؤال دون استطراد خارج الموضوع.
2. حجم التقرير متوازن ومركز (بين 150 إلى 350 كلمة كحد أقصى).
3. استخدام 2 إلى 3 أقسام رئيسية بعناوين فرعية واضحة ومنظمة.
4. كل قسم يحتوي على 2 إلى 3 جمل وشرح موجز، مع نقاط محددة (أقصى حد 3 إلى 4 نقاط).
5. تدعيم كل ادعاء بالأدلة الواردة من نتائج البحث.
6. إنهاء التقرير بقسم روابط المصادر والمراجع المباشرة (2 إلى 3 روابط).
{f"ملاحظة جوهرية للمراجعة رقم {revision_count}: عليك مراعاة واستيعاب الملاحظات النقدية السابقة بدقة وتحسين التقرير بناءً عليها." if critique and critique != "None" else ""}"""

        user_prompt = f"""سؤال البحث: {question}

نتائج البحث الميداني المسترجعة:
{search_results}

ملاحظات التدقيق السابقة (إن وجدت):
{critique}

اكتب التقرير البحثي المتكامل باللغة العربية:"""

    else:
        system_prompt = f"""You are a senior executive technical research writer.
Your job is to synthesize search results into a concise, factual, well-structured research brief.

Strict Writing Guidelines:
1. Directly answer the core research question without tangential fluff.
2. Maintain an executive word count: strictly between 150 and 350 words.
3. Organize into 2 to 3 clearly titled sections.
4. Maximum 3 to 4 bullet points per section with high information density.
5. Ground all assertions firmly in the retrieved search evidence.
6. Conclude with a dedicated 'Sources' section listing 2 to 3 clean reference URLs.
{f"Crucial Directive for Revision #{revision_count}: Directly incorporate the Critic's feedback below to rectify all identified deficiencies." if critique and critique != "None" else ""}"""

        user_prompt = f"""Research Question: {question}

Retrieved Search Results:
{search_results}

Previous Critique / Deficiencies:
{critique}

Synthesize the final research report:"""

    draft_answer = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2
    )

    return {"draft_answer": draft_answer}
