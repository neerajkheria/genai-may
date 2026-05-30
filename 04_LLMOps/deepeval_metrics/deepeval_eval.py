"""
DeepEval Metrics — RAG pipeline evaluation
Use Case: Evaluate faithfulness, relevancy, and contextual precision
"""
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DISABLE_DEEPEVAL_TELEMETRY"] = "1"

from dotenv import load_dotenv; load_dotenv()

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import (FaithfulnessMetric, AnswerRelevancyMetric,
                                   ContextualPrecisionMetric, ContextualRecallMetric)
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("deepeval not installed. Run: pip install deepeval")

def run_rag_evaluation():
    if not DEEPEVAL_AVAILABLE:
        return
    
    # Sample RAG test cases for insurance QA
    test_cases = [
        LLMTestCase(
            input="What is the maximum sum insured under family floater plan?",
            actual_output="Under a family floater plan, the maximum sum insured available is typically Rs. 50 lakhs, covering all family members under a single policy.",
            retrieval_context=["Family floater health insurance plans offer coverage up to Rs. 50 lakhs. All family members share the sum insured. Premium is based on the oldest member's age."],
            expected_output="The maximum sum insured under a family floater plan is Rs. 50 lakhs.",
        ),
        LLMTestCase(
            input="Does term insurance cover critical illness?",
            actual_output="Standard term insurance does not cover critical illness. However, you can add a critical illness rider that provides a lump-sum payout upon diagnosis of specified illnesses.",
            retrieval_context=["Term life insurance provides death benefit only. Critical illness coverage is available as an add-on rider for an additional premium."],
            expected_output="Standard term insurance does not cover critical illness unless a rider is added.",
        ),
    ]
    
    metrics = [
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        ContextualPrecisionMetric(threshold=0.6, model="gpt-4o-mini"),
        ContextualRecallMetric(threshold=0.6, model="gpt-4o-mini"),
    ]
    
    print("Running DeepEval metrics...")
    for case in test_cases:
        for metric in metrics:
            print(f"Scoring {metric.__class__.__name__} for: {case.input[:50]}...")
            metric.measure(case)
            print(f"  Score: {metric.score:.3f} | Pass: {metric.success}")

if __name__ == "__main__":
    run_rag_evaluation()
