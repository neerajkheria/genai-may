"""
Module 5 — Persistent Memory with Mem0 + Vector DB
Use Case: Personalized insurance advisor that remembers user preferences across sessions
"""
import os
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Note: Install mem0ai and qdrant-client
# pip install mem0ai qdrant-client

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("mem0ai not installed. Run: pip install mem0ai")

SYSTEM_PROMPT = """You are Arjun, a personalized insurance advisor at TrustShield.
You remember each customer's preferences, risk profile, and past interactions.
Use the provided memory context to personalize every response."""

class PersonalizedAdvisor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        if MEM0_AVAILABLE:
            config = {
                "vector_store": {"provider": "qdrant",
                                 "config": {"host": "localhost", "port": 6333}},
                "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            }
            self.memory = Memory.from_config(config)
        else:
            self.memory = None

    def chat(self, user_id: str, message: str) -> str:
        context = ""
        if self.memory:
            memories = self.memory.search(message, user_id=user_id, limit=5)
            if memories.get("results"):
                context = "\n".join([f"- {m['memory']}" for m in memories["results"]])

        messages = [
            SystemMessage(content=f"{SYSTEM_PROMPT}\n\nKnown preferences for this customer:\n{context}" if context else SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
        response = self.llm.invoke(messages).content

        if self.memory:
            self.memory.add([{"role": "user", "content": message},
                             {"role": "assistant", "content": response}], user_id=user_id)
        return response

def run_demo():
    advisor = PersonalizedAdvisor()
    user_id = "customer_rajesh_001"

    # Session 1
    print("=== Session 1 ===")
    interactions = [
        "Hi, I prefer term life insurance with low premiums. I am 35, non-smoker.",
        "I have a risk-averse profile and want sum assured of at least 1 crore.",
    ]
    for msg in interactions:
        print(f"Customer: {msg}")
        print(f"Arjun: {advisor.chat(user_id, msg)}\n")

    # Session 2 (new session, memory persists)
    print("\n=== Session 2 (memory persists) ===")
    print("Customer: What plan would you recommend for me?")
    print(f"Arjun: {advisor.chat(user_id, 'What plan would you recommend for me?')}")

if __name__ == "__main__":
    run_demo()
