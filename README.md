# Enterprise GenAI Training Program — Codebase

A complete, GitHub-ready collection of Python code samples, app modules, and reference implementations from the **15-Day Enterprise AI Training Bootcamp** (Insurance & Banking Domain).

## Folder Structure

| Folder | Description |
|--------|-------------|
| `01_LangChain_Agents_Memory/` | LangChain agents, tool calling, ReAct, memory, Mem0 |
| `02_Multi_Agent_Systems/` | CrewAI, OpenAI Swarm, LangGraph feedback loops |
| `03_RAG_Systems/` | Agentic RAG, CRAG pipeline, FAISS vector store |
| `04_LLMOps/` | LangSmith evaluation, DeepEval metrics, CI/CD |
| `05_Guardrails_Security/` | Prompt injection, PII redaction, JWT auth, Guardrails AI |
| `06_Optimization/` | Redis caching, FastAPI SSE streaming, retry/fallback |
| `07_MCP_Server/` | MCP server, insurance tools, dynamic tool discovery |
| `08_Capstone_ProShield/` | Full ProShield CRAG + MCP capstone project |
| `09_Monitoring_Observability/` | Prometheus, New Relic, LangSmith tracing |
| `10_Infrastructure/` | EC2 setup, Docker Compose, Terraform snippets |

## Prerequisites

- Python 3.11+
- AWS EC2 Ubuntu 22.04 (or local)
- OpenAI API Key
- LangSmith API Key (optional, for tracing)

## Quick Start

```bash
git clone <your-repo-url>
cd GenAI_Training_Codebase
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Tech Stack

LangChain · LangGraph · LangSmith · OpenAI · FastAPI · FAISS · Redis · DeepEval · Guardrails AI · CrewAI · Mem0 · Prometheus · Docker
