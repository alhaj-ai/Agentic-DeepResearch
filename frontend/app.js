/**
 * Agentic-DeepResearch Frontend Client
 * Developed by Mohammed AL-Haj (@alhaj-ai)
 */

const API_URL = window.location.origin + "/research";
let currentReportMarkdown = "";

function setPrompt(text) {
    const input = document.getElementById("question-input");
    input.value = text;
    input.focus();
}

function isArabic(text) {
    const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/;
    return arabicPattern.test(text);
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setAgentState(agentId, state, customText) {
    const card = document.getElementById(`agent-${agentId}`);
    const badge = document.getElementById(`badge-${agentId}`);
    if (!card || !badge) return;

    card.classList.remove("running", "done");

    if (state === "running") {
        card.classList.add("running");
        badge.innerText = customText || "Active...";
    } else if (state === "done") {
        card.classList.add("done");
        badge.innerText = customText || "✓ Complete";
    } else {
        badge.innerText = customText || "Waiting";
    }
}

function resetPipeline() {
    ["planner", "searcher", "writer", "critic"].forEach(id => {
        setAgentState(id, "waiting");
    });
    document.getElementById("error-box").style.display = "none";
    document.getElementById("result-box").style.display = "none";
    document.getElementById("pipeline-section").style.display = "block";
    document.getElementById("pipeline-status").innerText = "Executing LangGraph...";
}

function markdownToHtml(md) {
    if (!md) return "";
    let html = md
        // Escape basic HTML
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        // Headings
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        // Unordered lists
        .replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>')
        .replace(/^\s*\*\s+(.*$)/gim, '<li>$1</li>')
        // Numbered lists
        .replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        // Paragraphs & Line breaks
        .replace(/\n\n+/g, '</p><p>')
        .replace(/\n/g, '<br/>');

    // Wrap in paragraph
    html = `<p>${html}</p>`;
    // Clean empty paragraphs
    html = html.replace(/<p><\/p>/g, '');
    return html;
}

async function submitQuestion() {
    const input = document.getElementById("question-input");
    const question = input.value.trim();

    if (!question) {
        showError("Please enter a research topic or question.");
        return;
    }

    const isAr = isArabic(question);
    const submitBtn = document.getElementById("submit-btn");
    submitBtn.disabled = true;
    document.getElementById("btn-text").innerText = "Researching...";

    resetPipeline();

    // Visual sequence animation while server completes LangGraph nodes
    setAgentState("planner", "running", "Decomposing...");
    const plannerTimer = setTimeout(() => {
        setAgentState("planner", "done");
        setAgentState("searcher", "running", "Retrieving...");
    }, 1200);

    const searcherTimer = setTimeout(() => {
        setAgentState("searcher", "done");
        setAgentState("writer", "running", "Synthesizing...");
    }, 2800);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        clearTimeout(plannerTimer);
        clearTimeout(searcherTimer);

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server responded with code ${response.status}`);
        }

        const data = await response.json();

        // Complete all visual steps
        setAgentState("planner", "done");
        setAgentState("searcher", "done");
        setAgentState("writer", "done");
        setAgentState("critic", "running", "Reviewing...");
        await delay(500);
        setAgentState("critic", "done", data.is_satisfactory ? "✓ Verified" : "Revised");

        document.getElementById("pipeline-status").innerText = "Finished";

        displayReport(data, isAr);

    } catch (err) {
        clearTimeout(plannerTimer);
        clearTimeout(searcherTimer);
        showError(err.message || "An unexpected error occurred during research execution.");
    } finally {
        submitBtn.disabled = false;
        document.getElementById("btn-text").innerText = "Execute Research";
    }
}

function displayReport(data, isAr) {
    currentReportMarkdown = data.answer;

    const resultBox = document.getElementById("result-box");
    const answerContainer = document.getElementById("result-answer");
    const langTag = document.getElementById("lang-tag");
    const revisionsTag = document.getElementById("revisions-tag");

    // Direction and language tag
    if (data.language === "ar" || isAr) {
        resultBox.setAttribute("dir", "rtl");
        langTag.innerText = "🇾🇪/🇸🇦 العربية";
    } else {
        resultBox.setAttribute("dir", "ltr");
        langTag.innerText = "🌐 English";
    }

    revisionsTag.innerText = `Revisions: ${data.revisions}`;

    // Sub-questions
    const subQBox = document.getElementById("sub-questions-box");
    const subQList = document.getElementById("sub-questions-list");
    subQList.innerHTML = "";
    if (data.sub_questions && data.sub_questions.length > 0) {
        data.sub_questions.forEach(q => {
            const li = document.createElement("li");
            li.textContent = q;
            subQList.appendChild(li);
        });
        subQBox.style.display = "block";
    } else {
        subQBox.style.display = "none";
    }

    // Answer HTML
    answerContainer.innerHTML = markdownToHtml(data.answer);

    // Sources list
    const sourcesBox = document.getElementById("sources-box");
    const sourcesList = document.getElementById("sources-list");
    sourcesList.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach(url => {
            const li = document.createElement("li");
            const a = document.createElement("a");
            a.href = url;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.textContent = url;
            li.appendChild(a);
            sourcesList.appendChild(li);
        });
        sourcesBox.style.display = "block";
    } else {
        sourcesBox.style.display = "none";
    }

    resultBox.style.display = "block";
    resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(msg) {
    const errorBox = document.getElementById("error-box");
    const errorMsg = document.getElementById("error-message");
    errorMsg.textContent = msg;
    errorBox.style.display = "flex";
}

function copyReport() {
    if (!currentReportMarkdown) return;
    navigator.clipboard.writeText(currentReportMarkdown).then(() => {
        alert("Research report copied to clipboard in Markdown format!");
    }).catch(() => {
        alert("Could not copy automatically. Please select text manually.");
    });
}

function downloadReport() {
    if (!currentReportMarkdown) return;
    const blob = new Blob([currentReportMarkdown], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deep_research_report_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
