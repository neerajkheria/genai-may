"""
Retry & Fallback Strategies for LLM APIs
Circuit breaker, exponential backoff, and model cascade
"""
import time, random
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv; load_dotenv()
import openai

# ── Tenacity Retry Decorator ──────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    reraise=True,
)
def call_llm_with_retry(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM with automatic retry on rate limit / timeout errors."""
    llm = ChatOpenAI(model=model, temperature=0, request_timeout=30)
    return llm.invoke(prompt).content

# ── Model Cascade Fallback ─────────────────────────────────────────────────────
class ModelCascade:
    """Try models in order, falling back to the next on failure."""
    def __init__(self):
        self.models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo"]
        self.timeouts = [60, 45, 30]

    def invoke(self, prompt: str) -> dict:
        for i, (model, timeout) in enumerate(zip(self.models, self.timeouts)):
            try:
                llm = ChatOpenAI(model=model, temperature=0, request_timeout=timeout)
                response = llm.invoke(prompt)
                return {"content": response.content, "model_used": model, "attempt": i + 1}
            except Exception as e:
                print(f"Model {model} failed: {e}. Trying next...")
                if i == len(self.models) - 1:
                    return {"content": "Service temporarily unavailable. Please try again.",
                            "model_used": "fallback", "attempt": i + 1, "error": str(e)}

# ── Circuit Breaker ────────────────────────────────────────────────────────────
class CircuitBreaker:
    CLOSED = "CLOSED"; OPEN = "OPEN"; HALF_OPEN = "HALF_OPEN"
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.HALF_OPEN
            else:
                raise Exception("Circuit OPEN — service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED; self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise

if __name__ == "__main__":
    cascade = ModelCascade()
    result = cascade.invoke("What is a deductible in health insurance?")
    print(f"Model: {result['model_used']} | Response: {result['content'][:100]}")
