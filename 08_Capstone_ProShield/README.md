# ProShield Commercial Insurance AI Copilot

**Pattern:** CRAG (Corrective RAG) + MCP Server + Guardrails
**Domain:** Commercial Insurance (EL, PI, D&O, Business Interruption)

## Architecture

```
Streamlit UI → FastAPI → CRAG Agent Pipeline → FAISS Vectorstore
                    ↓
               MCP Tools (Risk Score, Premium, Regulatory)
                    ↓
               LangSmith Tracing + DeepEval Scoring
```

## Key Files

| File | Purpose |
|------|---------|
| `app/config.py` | Lazy-loaded settings (no module-level env vars) |
| `app/pipeline.py` | CRAG pipeline with LangGraph |
| `app/main.py` | FastAPI application entry point |

## Critical Pattern: Lazy Loading

All settings and LLM clients must be initialized **inside functions**,
never at module level. This prevents `ValidationError` on import.

```python
# WRONG — causes ValidationError
llm = ChatOpenAI(api_key=get_settings().openai_api_key)  # module level

# CORRECT — lazy initialization
def get_llm():
    return ChatOpenAI(api_key=get_settings().openai_api_key)  # inside function
```

## Bugs Fixed During Development

1. `ValidationError: openai_api_key required` → Move `get_settings()` inside functions
2. `Extra inputs not permitted` → Add `extra = "ignore"` to Pydantic Config
3. `ChatOpenAI instantiated at module level` → Wrap in lazy getter functions
4. `ImportError: duckduckgo-search` → Move `DuckDuckGoSearchRun()` inside function
