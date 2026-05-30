"""
Redis Caching for LLM Responses
Reduce latency and API costs with semantic caching
"""
import hashlib, json, os
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Install: pip install redis")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour

class CachedLLM:
    def __init__(self, model: str = "gpt-4o-mini", cache_ttl: int = CACHE_TTL):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.cache_ttl = cache_ttl
        self.redis_client = redis.from_url(REDIS_URL) if REDIS_AVAILABLE else None
        self.hits = 0; self.misses = 0

    def _cache_key(self, prompt: str) -> str:
        return f"llm_cache:{hashlib.md5(prompt.encode()).hexdigest()}"

    def invoke(self, prompt: str) -> dict:
        cache_key = self._cache_key(prompt)
        
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    self.hits += 1
                    return {"content": json.loads(cached)["content"], "cached": True, "cache_hit": True}
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Cache miss — call LLM
        self.misses += 1
        response = self.llm.invoke(prompt)
        
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps({"content": response.content}))
            except Exception as e:
                print(f"Cache write error: {e}")
        
        return {"content": response.content, "cached": False, "cache_hit": False}

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": f"{(self.hits/total*100):.1f}%" if total > 0 else "N/A"}

if __name__ == "__main__":
    llm = CachedLLM()
    questions = [
        "What is a deductible in health insurance?",
        "What is a deductible in health insurance?",  # Should hit cache
        "What is a co-payment in insurance?",
    ]
    for q in questions:
        result = llm.invoke(q)
        print(f"{'[CACHED]' if result['cached'] else '[LLM   ]'} {q[:50]}...")
    print(f"Stats: {llm.stats()}")
