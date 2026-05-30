"""
LangSmith Tracing Setup
Configure tracing for agents, RAG pipelines, and custom functions
"""
import os
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.traceable import traceable
from langsmith import Client, traceable as ls_traceable

# Enable tracing via environment variables
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "genai-bootcamp")

@traceable(name="insurance_rag_query", tags=["rag", "insurance"])
def rag_query(question: str, context: str) -> str:
    """Traced RAG query — visible in LangSmith dashboard."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Answer based on context only.\nContext: {context}\nQuestion: {question}"
    return llm.invoke(prompt).content

@traceable(name="claim_classification", tags=["classification"])
def classify_claim(claim_text: str) -> dict:
    """Classify insurance claim type and urgency."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(
        f"Classify this insurance claim. Return JSON: {{type, urgency, key_facts}}\n\nClaim: {claim_text}"
    )
    import json
    try:
        return json.loads(result.content)
    except:
        return {"type": "Unknown", "urgency": "Medium", "key_facts": result.content}

def get_run_url(run_id: str) -> str:
    """Get LangSmith URL for a specific run."""
    project = os.getenv("LANGCHAIN_PROJECT", "genai-bootcamp")
    return f"https://smith.langchain.com/o/your-org/projects/p/{project}/r/{run_id}"

if __name__ == "__main__":
    result = rag_query(
        question="What does business interruption insurance cover?",
        context="Business interruption insurance covers lost income and operating expenses when a business is forced to close due to a covered event like fire or flood."
    )
    print(f"RAG Result: {result}")
    
    claim = "Customer reports car damaged in parking lot collision. Rear bumper cracked. Photos attached."
    classification = classify_claim(claim)
    print(f"Claim Classification: {classification}")
