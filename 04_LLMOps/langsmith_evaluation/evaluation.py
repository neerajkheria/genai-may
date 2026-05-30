"""
LangSmith Evaluation — Agent output scoring and tracing
Use Case: Evaluate customer support agent on insurance queries
"""
import os
from dotenv import load_dotenv; load_dotenv()
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.traceable import traceable

@traceable(name="insurance_support_agent", project_name="genai-bootcamp")
def run_support_agent(query: str) -> str:
    """Simple agent for evaluation testing."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = "You are an insurance support specialist. Answer clearly and accurately."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    return llm.invoke(messages).content

def create_evaluation_dataset():
    """Create a LangSmith dataset for structured evaluation."""
    client = Client()
    dataset_name = "insurance-support-qa-v1"
    
    examples = [
        {"inputs": {"query": "What is the waiting period for pre-existing conditions?"},
         "outputs": {"answer": "Typically 2-4 years depending on the insurer and policy type."}},
        {"inputs": {"query": "How do I file a cashless claim at a hospital?"},
         "outputs": {"answer": "Visit hospital TPA desk, show insurance card, complete pre-authorization form."}},
        {"inputs": {"query": "What is the grace period for premium payment?"},
         "outputs": {"answer": "Most health policies offer 15-30 days grace period after due date."}},
    ]
    
    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
        for ex in examples:
            client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id)
        print(f"Dataset created: {dataset_name}")
        return dataset_name
    except Exception as e:
        print(f"Dataset may already exist: {e}")
        return dataset_name

def run_evaluation(dataset_name: str):
    """Run evaluation using LangSmith."""
    def predict(inputs: dict) -> dict:
        return {"answer": run_support_agent(inputs["query"])}
    
    evaluators = [
        LangChainStringEvaluator("qa"),
        LangChainStringEvaluator("context_qa"),
    ]
    
    results = evaluate(predict, data=dataset_name, evaluators=evaluators,
                       experiment_prefix="insurance-agent-eval")
    return results

if __name__ == "__main__":
    print("Testing agent...")
    response = run_support_agent("What documents are needed for a health claim?")
    print(f"Response: {response}")
