"""
Module 1 — LangChain Claim Validation Agent
Use Case: SilverShield Health Insurance — validate incoming claims
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

load_dotenv()

@tool
def validate_policy(policy_id: str) -> dict:
    """Validate whether a policy ID is active and in good standing."""
    mock_policies = {
        "POL-1001": {"status": "active", "holder": "Rajesh Kumar", "sum_assured": 500000},
        "POL-1002": {"status": "lapsed", "holder": "Priya Singh", "sum_assured": 300000},
    }
    policy = mock_policies.get(policy_id)
    if not policy:
        return {"valid": False, "reason": "Policy ID not found"}
    if policy["status"] != "active":
        return {"valid": False, "reason": f"Policy is {policy['status']}"}
    return {"valid": True, **policy}

@tool
def check_fraud_indicators(claim_amount: float, diagnosis_code: str, provider_id: str) -> dict:
    """Check claim for known fraud patterns and risk signals."""
    risk_score = 0
    flags = []
    if claim_amount > 200000:
        risk_score += 30; flags.append("High claim amount")
    if diagnosis_code in ["Z99.9", "T14.91"]:
        risk_score += 40; flags.append("Suspicious diagnosis code")
    if provider_id in ["PROV-999", "PROV-888"]:
        risk_score += 50; flags.append("Provider flagged for fraud")
    risk_level = "HIGH" if risk_score >= 60 else "MEDIUM" if risk_score >= 30 else "LOW"
    return {"risk_score": risk_score, "risk_level": risk_level, "flags": flags}

@tool
def estimate_coverage(policy_id: str, procedure_code: str, claim_amount: float) -> dict:
    """Calculate covered amount based on policy terms."""
    covered = claim_amount * 0.80
    deductible = 5000.0
    payable = max(0, covered - deductible)
    return {"claim_amount": claim_amount, "covered_amount": covered,
            "deductible": deductible, "net_payable": payable}

SYSTEM_PROMPT = """You are a senior claims validation officer at SilverShield Health Insurance.
Validate claims by: verifying policy, checking fraud, estimating coverage.
Recommend: APPROVE, REJECT, or ESCALATE."""

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [validate_policy, check_fraud_indicators, estimate_coverage]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(query: str, chat_history: list = None) -> str:
    result = build_agent().invoke({"input": query, "chat_history": chat_history or []})
    return result["output"]

if __name__ == "__main__":
    response = run_agent("""
    Validate this claim:
    - Policy ID: POL-1001
    - Claim Amount: Rs. 85,000
    - Diagnosis Code: J18.9 (Pneumonia)
    - Provider ID: PROV-101
    - Procedure Code: PROC-HOSP-ACUTE
    """)
    print(response)
