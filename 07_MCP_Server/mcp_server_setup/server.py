"""
MCP (Model Context Protocol) Server for Insurance Tools
Exposes tools dynamically to AI agents via MCP protocol
"""
import json
from dotenv import load_dotenv; load_dotenv()

# Install: pip install mcp
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Install: pip install mcp")

# ── Tool Implementations ───────────────────────────────────────────────────────
def query_policy_database(policy_id: str, query_type: str = "coverage") -> dict:
    """Query insurance policy database for coverage and terms."""
    policies = {
        "POL-1001": {"holder": "Rajesh Kumar", "type": "Health", "sum_assured": 500000,
                     "coverage": ["Hospitalization", "Surgery", "Day Care"], "exclusions": ["Cosmetic", "Self-harm"]},
        "POL-2001": {"holder": "Priya Ventures Ltd", "type": "Commercial", "sum_assured": 5000000,
                     "coverage": ["Property", "Liability", "Business Interruption"], "exclusions": ["War", "Nuclear"]},
    }
    policy = policies.get(policy_id)
    if not policy:
        return {"error": f"Policy {policy_id} not found"}
    if query_type == "coverage":
        return {"policy_id": policy_id, "coverage": policy["coverage"], "sum_assured": policy["sum_assured"]}
    return policy

def fetch_claim_status(claim_id: str) -> dict:
    """Fetch real-time claim processing status."""
    claims = {
        "CLM-5001": {"status": "Under Review", "amount": 85000, "days_pending": 3, "next_action": "Medical records review"},
        "CLM-5002": {"status": "Approved", "amount": 45000, "settlement_date": "2025-12-01", "utr": "UTR-ABC123"},
        "CLM-5003": {"status": "Rejected", "amount": 150000, "reason": "Pre-existing condition exclusion"},
    }
    claim = claims.get(claim_id)
    return claim if claim else {"error": f"Claim {claim_id} not found"}

def filter_policies_by_metadata(product_type: str, min_coverage: int = 0, max_premium: int = None) -> list:
    """Filter policies by metadata like type, coverage range, premium range."""
    all_policies = [
        {"id": "PROD-101", "type": "Health", "coverage": 500000, "annual_premium": 12000, "name": "HealthShield Basic"},
        {"id": "PROD-102", "type": "Health", "coverage": 1000000, "annual_premium": 18000, "name": "HealthShield Plus"},
        {"id": "PROD-201", "type": "Term", "coverage": 10000000, "annual_premium": 8000, "name": "LifeSecure Term 30"},
        {"id": "PROD-301", "type": "Commercial", "coverage": 5000000, "annual_premium": 45000, "name": "BizShield Pro"},
    ]
    filtered = [p for p in all_policies
                if p["type"].lower() == product_type.lower()
                and p["coverage"] >= min_coverage
                and (max_premium is None or p["annual_premium"] <= max_premium)]
    return filtered

if MCP_AVAILABLE:
    server = Server("insurance-mcp-server")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(name="query_policy_database",
                       description="Query insurance policy database for coverage, terms, and exclusions",
                       inputSchema={"type": "object",
                                    "properties": {"policy_id": {"type": "string"},
                                                   "query_type": {"type": "string", "enum": ["coverage", "full"]}},
                                    "required": ["policy_id"]}),
            types.Tool(name="fetch_claim_status",
                       description="Fetch real-time insurance claim processing status",
                       inputSchema={"type": "object",
                                    "properties": {"claim_id": {"type": "string"}},
                                    "required": ["claim_id"]}),
            types.Tool(name="filter_policies_by_metadata",
                       description="Filter and search insurance products by type and coverage",
                       inputSchema={"type": "object",
                                    "properties": {"product_type": {"type": "string"},
                                                   "min_coverage": {"type": "integer"},
                                                   "max_premium": {"type": "integer"}},
                                    "required": ["product_type"]}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "query_policy_database":
            result = query_policy_database(**arguments)
        elif name == "fetch_claim_status":
            result = fetch_claim_status(**arguments)
        elif name == "filter_policies_by_metadata":
            result = filter_policies_by_metadata(**arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    if __name__ == "__main__":
        import asyncio
        asyncio.run(main())
else:
    print("MCP server stub — install mcp package to run")
    if __name__ == "__main__":
        # Demo mode: test tool functions directly
        print(json.dumps(query_policy_database("POL-1001", "coverage"), indent=2))
        print(json.dumps(fetch_claim_status("CLM-5001"), indent=2))
        print(json.dumps(filter_policies_by_metadata("Health", min_coverage=500000), indent=2))
