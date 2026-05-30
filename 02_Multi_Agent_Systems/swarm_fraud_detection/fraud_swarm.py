"""
OpenAI Swarm — Fraud Detection Pipeline
Use Case: 4-agent pipeline for transaction fraud analysis
Agents: Transaction Receiver → Rule Screener → ML Risk Scorer → Fraud Decision
"""
# Install: pip install git+https://github.com/openai/swarm.git
try:
    from swarm import Swarm, Agent
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    print("Swarm not installed. Run: pip install git+https://github.com/openai/swarm.git")

from dotenv import load_dotenv; load_dotenv()

# ── Handoff Functions ─────────────────────────────────────────────────────────
def handoff_to_rule_screener():
    return rule_screener_agent

def handoff_to_ml_scorer():
    return ml_scorer_agent

def handoff_to_fraud_decision():
    return fraud_decision_agent

# ── Agent Definitions ─────────────────────────────────────────────────────────
if SWARM_AVAILABLE:
    transaction_receiver = Agent(
        name="Transaction Receiver",
        instructions="""Parse the incoming transaction and extract:
        amount, currency, merchant, country, timestamp, card_type.
        Standardize the data and pass to the Rule Screener.""",
        functions=[handoff_to_rule_screener],
    )

    rule_screener_agent = Agent(
        name="Rule-Based Screener",
        instructions="""Apply rule-based fraud checks:
        - Flag if amount > Rs. 50,000 in a single transaction
        - Flag if country is in high-risk list (NG, KP, IR)
        - Flag if transaction time is 2AM-5AM IST
        - Flag if 3+ transactions in last 10 minutes
        
        Score: 0-30=LOW, 31-60=MEDIUM, 61-100=HIGH
        Pass results to ML Risk Scorer.""",
        functions=[handoff_to_ml_scorer],
    )

    ml_scorer_agent = Agent(
        name="ML Risk Scorer",
        instructions="""Simulate ML model scoring (in production this calls a model endpoint).
        Consider: velocity patterns, merchant category risk, geographic anomalies,
        behavioral deviation from customer history.
        
        Combine with rule-based score using weights:
        Final Score = 0.4 * rule_score + 0.6 * ml_score
        Pass final score to Fraud Decision agent.""",
        functions=[handoff_to_fraud_decision],
    )

    fraud_decision_agent = Agent(
        name="Fraud Decision Officer",
        instructions="""Make final fraud determination:
        - Score 0-30: APPROVE transaction
        - Score 31-60: FLAG for 2FA verification
        - Score 61-100: BLOCK and alert customer
        
        Provide: decision, score, reason, recommended_action.""",
    )

def analyze_transaction(transaction: dict) -> str:
    if not SWARM_AVAILABLE:
        return "Swarm not available. Install: pip install git+https://github.com/openai/swarm.git"
    
    client = Swarm()
    message = f"""Analyze this transaction for fraud:
    Amount: {transaction.get("amount")}
    Merchant: {transaction.get("merchant")}
    Country: {transaction.get("country")}
    Time: {transaction.get("timestamp")}
    Card Type: {transaction.get("card_type", "Credit")}
    """
    response = client.run(
        agent=transaction_receiver,
        messages=[{"role": "user", "content": message}],
        context_variables={"transaction": transaction},
    )
    return response.messages[-1]["content"]

if __name__ == "__main__":
    high_risk_txn = {"amount": 75000, "merchant": "Electronics Store", "country": "NG", 
                     "timestamp": "2025-11-15 03:15:00", "card_type": "Credit"}
    normal_txn = {"amount": 2500, "merchant": "Grocery Store", "country": "IN", 
                  "timestamp": "2025-11-15 14:30:00", "card_type": "Debit"}
    
    print("=== HIGH RISK TRANSACTION ===")
    print(analyze_transaction(high_risk_txn))
    print("\n=== NORMAL TRANSACTION ===")
    print(analyze_transaction(normal_txn))
