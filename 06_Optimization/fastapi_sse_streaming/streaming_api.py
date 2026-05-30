"""
FastAPI Server-Sent Events (SSE) Streaming
Real-time streaming LLM responses to clients
"""
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncIteratorCallbackHandler
from dotenv import load_dotenv; load_dotenv()

app = FastAPI(title="Streaming Insurance Advisor API")

class StreamRequest(BaseModel):
    query: str
    session_id: str = "default"

async def stream_llm_response(query: str):
    """Generator that yields SSE-formatted LLM tokens."""
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True,
                     callbacks=[callback])
    
    task = asyncio.create_task(
        llm.ainvoke([{"role": "system", "content": "You are a helpful insurance advisor."},
                     {"role": "user", "content": query}])
    )
    
    try:
        async for token in callback.aiter():
            yield f"data: {token}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {str(e)}\n\n"
    finally:
        await task
    
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """SSE endpoint — streams tokens as they are generated."""
    return StreamingResponse(
        stream_llm_response(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# Client example (run in separate terminal):
# curl -N -X POST http://localhost:8000/chat/stream \
#   -H "Content-Type: application/json" \
#   -d '{"query": "Explain term insurance in simple terms"}'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
