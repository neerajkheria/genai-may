"""
ProShield CRAG Pipeline — Main orchestration
Self-Reflective RAG with compliance guardrails
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def get_llm():
    """Lazy LLM initialization — never call at module level."""
    from app.config import get_settings
    settings = get_settings()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                      openai_api_key=settings.openai_api_key)

def get_web_search():
    """Lazy web search initialization."""
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun()

class PipelineState(TypedDict):
    question: str
    documents: list
    generation: str
    grade: str          # "relevant" | "irrelevant"
    rewrite_count: int
    compliance_passed: bool

def retrieve(state: PipelineState, vectorstore: FAISS) -> PipelineState:
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {**state, "documents": docs}

def grade_documents(state: PipelineState) -> PipelineState:
    llm = get_llm()
    question = state["question"]
    grades = []
    for doc in state["documents"]:
        result = llm.invoke(
            f"Is this document relevant to the question? Answer only yes/no.\n"
            f"Question: {question}\nDocument: {doc.page_content[:500]}"
        )
        grades.append("yes" in result.content.lower())
    grade = "relevant" if sum(grades) > len(grades) / 2 else "irrelevant"
    return {**state, "grade": grade}

def generate(state: PipelineState) -> PipelineState:
    llm = get_llm()
    context = "\n\n".join([d.page_content for d in state["documents"]])
    response = llm.invoke(
        f"Answer this insurance question based on the context only.\n"
        f"Context: {context}\n\nQuestion: {state['question']}"
    )
    return {**state, "generation": response.content}

def rewrite_query(state: PipelineState) -> PipelineState:
    llm = get_llm()
    result = llm.invoke(f"Rewrite this question to improve document search:\n{state['question']}")
    return {**state, "question": result.content, "rewrite_count": state.get("rewrite_count", 0) + 1}

def compliance_check(state: PipelineState) -> PipelineState:
    llm = get_llm()
    result = llm.invoke(
        f"Does this response contain PII, illegal advice, or hallucinated facts? Answer only yes/no.\n"
        f"Response: {state['generation']}"
    )
    passed = "no" in result.content.lower()
    return {**state, "compliance_passed": passed,
            "generation": state["generation"] if passed else "[Response blocked by compliance guardrail]"}

def route_after_grade(state: PipelineState) -> str:
    if state["grade"] == "relevant":
        return "generate"
    if state.get("rewrite_count", 0) >= 2:
        return "generate"
    return "rewrite"

def build_pipeline(vectorstore: FAISS):
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("retrieve", lambda s: retrieve(s, vectorstore))
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("compliance", compliance_check)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_after_grade,
                                   {"generate": "generate", "rewrite": "rewrite"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "compliance")
    workflow.add_edge("compliance", END)
    
    return workflow.compile()
