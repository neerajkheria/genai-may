"""
LangGraph Agent Feedback Loop
Use Case: Insurance appeal letter — Planner writes, Validator reviews, loop until approved
"""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv; load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── State Schema ──────────────────────────────────────────────────────────────
class AppealState(TypedDict):
    claim_details: str          # Original claim info
    proposal: str               # Current draft from planner
    feedback: str               # Validator's feedback
    iteration_count: int        # Loop counter (max 3)
    approved: bool              # Terminal flag
    final_letter: str           # Approved output

# ── Node Functions ────────────────────────────────────────────────────────────
def planner_node(state: AppealState) -> AppealState:
    """Write or revise the appeal letter based on feedback."""
    if state["feedback"]:
        prompt = f"""Revise this insurance appeal letter based on the validator feedback.
        
Original claim: {state["claim_details"]}
Previous draft: {state["proposal"]}
Validator feedback: {state["feedback"]}

Write an improved version that addresses all the issues raised."""
    else:
        prompt = f"""Write a professional insurance claim appeal letter for:
{state["claim_details"]}

The letter should be formal, factual, and include specific policy references."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "proposal": response.content, "iteration_count": state["iteration_count"] + 1}

def validator_node(state: AppealState) -> AppealState:
    """Review the appeal letter and provide structured feedback."""
    prompt = f"""Review this insurance appeal letter and evaluate it on:
1. Professional tone and language
2. Factual accuracy and completeness  
3. Clear articulation of the appeal grounds
4. Proper policy references
5. Call to action

Appeal letter:
{state["proposal"]}

Respond with either:
- "APPROVED: [brief reason]" if the letter meets all criteria
- "REJECTED: [specific issues to fix]" if improvements are needed"""

    response = llm.invoke([HumanMessage(content=prompt)])
    feedback_text = response.content
    approved = feedback_text.upper().startswith("APPROVED")
    
    return {**state, 
            "feedback": feedback_text,
            "approved": approved,
            "final_letter": state["proposal"] if approved else ""}

def should_continue(state: AppealState) -> str:
    """Conditional edge: approve or keep looping."""
    if state["approved"]:
        return "end"
    if state["iteration_count"] >= 3:
        return "end"  # Max iterations reached
    return "planner"  # Revise and retry

def build_feedback_graph():
    workflow = StateGraph(AppealState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("validator", validator_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "validator")
    workflow.add_conditional_edges("validator", should_continue,
                                   {"end": END, "planner": "planner"})
    return workflow.compile()

def generate_appeal_letter(claim_details: str) -> dict:
    graph = build_feedback_graph()
    initial_state = AppealState(
        claim_details=claim_details, proposal="", feedback="",
        iteration_count=0, approved=False, final_letter=""
    )
    final_state = graph.invoke(initial_state)
    return {"approved": final_state["approved"], "iterations": final_state["iteration_count"],
            "letter": final_state["final_letter"] or final_state["proposal"],
            "final_feedback": final_state["feedback"]}

if __name__ == "__main__":
    claim = """
    Policy: SH-HEALTH-5523 | Holder: Mr. Prakash Nair
    Claim rejected for: Cataract surgery, Rs. 95,000
    Rejection reason: Classified as pre-existing condition
    Facts: Diagnosis made only 6 months ago; policy is 3 years old
    """
    result = generate_appeal_letter(claim)
    print(f"Approved: {result['approved']} | Iterations: {result['iterations']}")
    print("\n=== FINAL LETTER ===")
    print(result["letter"])
