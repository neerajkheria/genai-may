"""
Prometheus Metrics for FastAPI GenAI App
Tracks: request count, latency, token usage, error rate
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
import time
from dotenv import load_dotenv; load_dotenv()

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Install: pip install prometheus-client")

app = FastAPI()

if PROMETHEUS_AVAILABLE:
    # ── Metrics Definitions ───────────────────────────────────────────────────
    REQUEST_COUNT = Counter("genai_requests_total", "Total requests", ["method", "endpoint", "status"])
    REQUEST_LATENCY = Histogram("genai_request_duration_seconds", "Request latency",
                                ["endpoint"], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    TOKEN_USAGE = Counter("genai_tokens_total", "Total tokens used", ["model", "type"])
    ACTIVE_REQUESTS = Gauge("genai_active_requests", "Currently active requests")
    ERROR_COUNT = Counter("genai_errors_total", "Total errors", ["error_type"])

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
            REQUEST_COUNT.labels(method=request.method,
                                  endpoint=request.url.path,
                                  status=response.status_code).inc()
            return response
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    if PROMETHEUS_AVAILABLE:
        TOKEN_USAGE.labels(model="gpt-4o-mini", type="prompt").inc(100)
        TOKEN_USAGE.labels(model="gpt-4o-mini", type="completion").inc(50)
    return {"response": f"Processed: {body.get('query', '')}", "tokens_used": 150}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
