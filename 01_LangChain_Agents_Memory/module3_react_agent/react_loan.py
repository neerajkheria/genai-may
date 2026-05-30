"""
Module 3 — ReAct Agent for Loan Eligibility Reasoning
Use Case: NovaNorth Bank — assess loan applications using FOIR
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv; load_dotenv()

@tool
def calculate_foir(monthly_income: float, existing_emis: float, proposed_emi: float) -> dict:
    """Calculate Fixed Obligation to Income Ratio (FOIR) for loan eligibility."""
    total_obligations = existing_emis + proposed_emi
    foir = (total_obligations / monthly_income) * 100
    eligible = foir <= 50
    return {"monthly_income": monthly_income, "total_obligations": total_obligations,
            "foir_percent": round(foir, 2), "eligible": eligible,
            "max_allowed_foir": 50, "recommendation": "ELIGIBLE" if eligible else "REJECT - FOIR too high"}

@tool
def check_credit_score(score: int) -> dict:
    """Evaluate credit score for loan eligibility."""
    if score >= 750: category = "Excellent"; eligible = True; rate = 8.5
    elif score >= 700: category = "Good"; eligible = True; rate = 10.0
    elif score >= 650: category = "Fair"; eligible = True; rate = 13.5
    else: category = "Poor"; eligible = False; rate = None
    return {"score": score, "category": category, "eligible": eligible, "interest_rate": rate}

@tool
def assess_employment(employment_type: str, years_experience: float, monthly_income: float) -> dict:
    """Assess employment stability for loan risk scoring."""
    stable_types = ["salaried_permanent", "government", "psu"]
    risk = "LOW" if employment_type in stable_types and years_experience >= 2 else "MEDIUM" if years_experience >= 1 else "HIGH"
    loan_multiplier = {"LOW": 60, "MEDIUM": 48, "HIGH": 36}[risk]
    max_loan = monthly_income * loan_multiplier
    return {"employment_type": employment_type, "risk_category": risk,
            "max_loan_amount": max_loan, "eligible": risk != "HIGH"}

@tool
def calculate_ltv(property_value: float, loan_amount: float) -> dict:
    """Calculate Loan-to-Value ratio for mortgage applications."""
    ltv = (loan_amount / property_value) * 100
    eligible = ltv <= 80
    return {"property_value": property_value, "loan_amount": loan_amount,
            "ltv_percent": round(ltv, 2), "max_allowed_ltv": 80, "eligible": eligible}

REACT_TEMPLATE = """You are a senior loan underwriter at NovaNorth Bank. 
Use the available tools to evaluate loan applications step by step.
Think through each factor: credit score, FOIR, employment stability, and LTV.

Tools available:
{tools}

Use this format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

def run_react_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculate_foir, check_credit_score, assess_employment, calculate_ltv]
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor.invoke({"input": query})["output"]

if __name__ == "__main__":
    result = run_react_agent("""
    Evaluate home loan application:
    - Applicant: Amit Verma
    - Monthly Income: Rs. 80,000
    - Existing EMIs: Rs. 15,000/month
    - Proposed Loan: Rs. 40,00,000 (EMI approx Rs. 35,000/month)
    - Credit Score: 720
    - Employment: Salaried permanent, 4 years experience
    - Property Value: Rs. 55,00,000
    """)
    print(result)
