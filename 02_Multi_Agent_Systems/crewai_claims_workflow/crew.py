"""
CrewAI Claims Processing Workflow
Use Case: SilverShield Health — multi-agent claims pipeline
Agents: Claims Intake → Policy Validator → Approval Officer
"""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv; load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Agents ────────────────────────────────────────────────────────────────────
intake_agent = Agent(
    role="Claims Intake Officer",
    goal="Extract and structure all relevant information from incoming insurance claims",
    backstory="You are an experienced claims intake specialist who ensures all required information is captured accurately before processing begins.",
    llm=llm, verbose=True, allow_delegation=False,
)

validator_agent = Agent(
    role="Policy Validation Specialist",
    goal="Verify policy coverage, check for exclusions, and assess claim validity against policy terms",
    backstory="You are a meticulous policy expert with deep knowledge of insurance terms and conditions. You identify coverage gaps and policy violations.",
    llm=llm, verbose=True, allow_delegation=False,
)

approval_agent = Agent(
    role="Senior Claims Approval Officer",
    goal="Make final approval decisions based on validated information and provide settlement amounts",
    backstory="You are a senior claims officer with authority to approve, reject, or escalate claims. You balance customer satisfaction with fraud prevention.",
    llm=llm, verbose=True, allow_delegation=True,
)

# ── Tasks ─────────────────────────────────────────────────────────────────────
def create_tasks(claim_data: str):
    intake_task = Task(
        description=f"""Process the following claim submission and extract structured information:
        {claim_data}
        
        Extract: Claimant name, Policy ID, Date of incident, Type of claim,
        Claimed amount, Hospital/Provider details, Supporting documents listed.""",
        agent=intake_agent,
        expected_output="Structured claim summary with all extracted fields",
    )

    validation_task = Task(
        description="""Review the claim summary from the intake officer and validate:
        1. Policy coverage for the claimed treatment/incident
        2. Check for common exclusions (pre-existing, waiting period, etc.)
        3. Verify claimed amount is within policy limits
        4. Flag any missing documents or information""",
        agent=validator_agent,
        expected_output="Validation report with coverage confirmation and any issues found",
        context=[intake_task],
    )

    approval_task = Task(
        description="""Based on the intake summary and validation report:
        1. Make a final decision: APPROVE / REJECT / ESCALATE
        2. If APPROVE: calculate exact settlement amount
        3. If REJECT: provide clear reason with policy reference
        4. If ESCALATE: explain what additional review is needed
        Provide a formal claims decision letter.""",
        agent=approval_agent,
        expected_output="Formal claims decision with settlement amount or rejection reason",
        context=[intake_task, validation_task],
    )
    return [intake_task, validation_task, approval_task]

def process_claim(claim_data: str) -> str:
    tasks = create_tasks(claim_data)
    crew = Crew(agents=[intake_agent, validator_agent, approval_agent],
                tasks=tasks, process=Process.sequential, memory=False, verbose=2)
    result = crew.kickoff()
    return str(result)

if __name__ == "__main__":
    sample_claim = """
    Claimant: Mrs. Anita Sharma
    Policy ID: SS-HEALTH-7823
    Date of Hospitalization: 15-Nov-2025
    Discharge Date: 22-Nov-2025 (7 days)
    Hospital: Apollo Hospital, Mumbai (NABH Accredited)
    Diagnosis: Appendicitis with Laparoscopic Surgery
    ICD Code: K35.9
    Total Bill: Rs. 1,85,000
    Room Type: Single AC (Rs. 8,000/day)
    Documents: Discharge Summary, Bills, Prescriptions submitted
    """
    result = process_claim(sample_claim)
    print("\n=== CREW DECISION ===")
    print(result)
