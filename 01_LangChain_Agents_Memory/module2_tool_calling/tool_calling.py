"""
Module 2 — Parallel Tool Calling with Pydantic-typed tools
Use Case: External API simulation for CIBIL, Hospital, RTO lookups
"""
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv; load_dotenv()

class CibilInput(BaseModel):
    pan_number: str = Field(description="PAN number of the applicant")

class HospitalInput(BaseModel):
    hospital_id: str = Field(description="Hospital registration ID")
    claim_amount: float = Field(description="Claimed amount in INR")

class RtoInput(BaseModel):
    vehicle_number: str = Field(description="Vehicle registration number")

def fetch_cibil_score(pan_number: str) -> dict:
    """Simulate CIBIL credit score API call."""
    mock = {"ABCDE1234F": {"score": 750, "status": "Good", "active_loans": 2},
            "XYZAB9876G": {"score": 480, "status": "Poor", "active_loans": 5}}
    return mock.get(pan_number, {"score": 650, "status": "Average", "active_loans": 1})

def verify_hospital(hospital_id: str, claim_amount: float) -> dict:
    """Verify hospital accreditation and claim eligibility."""
    accredited = {"HOSP-101": "NABH Accredited", "HOSP-202": "NABL Accredited"}
    status = accredited.get(hospital_id, "Not Accredited")
    eligible = claim_amount <= 500000 and status != "Not Accredited"
    return {"hospital_id": hospital_id, "accreditation": status,
            "claim_eligible": eligible, "max_cashless": 300000}

def lookup_vehicle(vehicle_number: str) -> dict:
    """Look up vehicle registration and insurance status."""
    mock = {"MH12AB1234": {"owner": "Rohit Sharma", "insured": True, "class": "Private Car"},
            "DL01CD5678": {"owner": "Sunita Devi", "insured": False, "class": "Two Wheeler"}}
    return mock.get(vehicle_number, {"owner": "Unknown", "insured": False})

cibil_tool = StructuredTool.from_function(func=fetch_cibil_score, name="fetch_cibil_score",
    description="Fetch CIBIL credit score for a PAN number", args_schema=CibilInput)
hospital_tool = StructuredTool.from_function(func=verify_hospital, name="verify_hospital",
    description="Verify hospital accreditation and claim eligibility", args_schema=HospitalInput)
rto_tool = StructuredTool.from_function(func=lookup_vehicle, name="lookup_vehicle",
    description="Look up vehicle registration details", args_schema=RtoInput)

def run_tool_agent(query: str) -> str:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [cibil_tool, hospital_tool, rto_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an insurance verification assistant. Use the available tools to gather information and provide a comprehensive assessment."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor.invoke({"input": query})["output"]

if __name__ == "__main__":
    result = run_tool_agent(
        "Check loan eligibility for PAN ABCDE1234F. "
        "Also verify hospital HOSP-101 for a Rs.1,50,000 claim. "
        "And look up vehicle MH12AB1234."
    )
    print(result)
