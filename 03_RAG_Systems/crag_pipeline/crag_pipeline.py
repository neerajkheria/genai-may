"""
CRAG (Corrective RAG) Pipeline
Use Case: ProShield Commercial Insurance — self-reflective document QA
"""
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv; load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# ── State Schema ──────────────────────────────────────────────────────────────
class CRAGState(TypedDict):
    question: str
    documents: list
    generation: str
    document_grade: str   # "relevant" | "irrelevant"
    rewrite_count: int
    web_search_used: bool

# ── Grader ────────────────────────────────────────────────────────────────────
def grade_documents(state: CRAGState) -> CRAGState:
    """Grade retrieved documents for relevance."""
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document relevance grader. Answer only 'relevant' or 'irrelevant'."),
        ("human", "Question: {question}\nDocument: {document}\nIs this document relevant?"),
    ])
    chain = grader_prompt | llm
    grades = []
    for doc in state["documents"]:
        result = chain.invoke({"question": state["question"], "document": doc.page_content})
        grades.append(result.content.lower().strip())
    
    overall = "relevant" if grades.count("relevant") > len(grades) / 2 else "irrelevant"
    return {**state, "document_grade": overall}

# ── Generator ─────────────────────────────────────────────────────────────────
def generate_answer(state: CRAGState) -> CRAGState:
    """Generate answer from retrieved context."""
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a commercial insurance expert. Answer based on the provided context only."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "generation": result.content}

# ── Rewriter ──────────────────────────────────────────────────────────────────
def rewrite_query(state: CRAGState) -> CRAGState:
    """Rewrite the query to improve retrieval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query optimizer. Rewrite the question to improve document retrieval."),
        ("human", "Original question: {question}\nRewrite it to be more specific and searchable."),
    ])
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    return {**state, "question": result.content, "rewrite_count": state.get("rewrite_count", 0) + 1}

# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_grading(state: CRAGState) -> str:
    if state["document_grade"] == "relevant":
        return "generate"
    if state.get("rewrite_count", 0) >= 2:
        return "generate"  # Generate with what we have after max rewrites
    return "rewrite"

def build_crag_graph(vectorstore: FAISS):
    def retrieve(state: CRAGState) -> CRAGState:
        docs = vectorstore.similarity_search(state["question"], k=4)
        return {**state, "documents": docs}

    workflow = StateGraph(CRAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("rewrite", rewrite_query)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_after_grading, {"generate": "generate", "rewrite": "rewrite"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()

def run_crag_demo():
    # Sample documents (in production these come from PDF loading)
    sample_docs = [
        Document(page_content="Commercial general liability insurance covers bodily injury and property damage claims arising from business operations. Standard limits are Rs.1 crore per occurrence.", metadata={"source": "CGL_Policy.pdf"}),
        Document(page_content="Professional indemnity insurance protects businesses against claims of negligence or breach of duty. Coverage includes legal defense costs up to policy limits.", metadata={"source": "PI_Policy.pdf"}),
        Document(page_content="Directors and Officers liability insurance covers personal liability of company directors for wrongful acts. Coverage applies to securities claims and employment practices.", metadata={"source": "DO_Policy.pdf"}),
    ]
    vectorstore = FAISS.from_documents(sample_docs, embeddings)
    graph = build_crag_graph(vectorstore)
    
    questions = [
        "What does commercial general liability cover?",
        "How does professional indemnity protect my business?",
    ]
    for q in questions:
        print(f"Q: {q}")
        result = graph.invoke({"question": q, "documents": [], "generation": "", 
                               "document_grade": "", "rewrite_count": 0, "web_search_used": False})
        print(f"A: {result['generation']}\n")

if __name__ == "__main__":
    run_crag_demo()
