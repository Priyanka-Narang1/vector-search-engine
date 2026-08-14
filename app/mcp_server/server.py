import asyncio
import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.coordinator.coordinator import Coordinator
from app.core.shard_config import SHARD_ADDRESSES


app = Server("vector-search-engine")
_coordinator = None


def _get_coordinator():
    # Lazy init - avoids connecting to shards/Redis at import time, only
    # when the tool is actually invoked. Matters because MCP servers get
    # imported/introspected by clients before any tool call happens.
    global _coordinator
    if _coordinator is None:
        _coordinator = Coordinator(SHARD_ADDRESSES)
    return _coordinator


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="semantic_search",
            description="Search the distributed vector index for items semantically similar to a query vector.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The query embedding vector.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top results to return.",
                        "default": 5,
                    },
                },
                "required": ["query_vector"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name, arguments):
    if name != "semantic_search":
        # Unknown tool name - fail loudly and specifically rather than a
        # generic error, so a misbehaving client sees exactly what went wrong.
        raise ValueError("Unknown tool: " + str(name))

    query_vector = arguments.get("query_vector")
    if not query_vector or not isinstance(query_vector, list):
        # Malformed input from an MCP client must not crash the server -
        # return a clear MCP-level error instead.
        return [TextContent(type="text", text="Error: query_vector must be a non-empty list of numbers")]

    k = arguments.get("k", 5)

    try:
        coordinator = _get_coordinator()
        result = coordinator.search(query_vector, k=k)
    except Exception as e:
        # Any downstream failure (shard/gRPC/etc) surfaces as a clean MCP
        # error response, not an unhandled exception killing the server.
        return [TextContent(type="text", text="Error running search: " + str(e))]

    lines = ["Top " + str(len(result["results"])) + " results:"]
    for r in result["results"]:
        lines.append("  id=" + str(r["id"]) + " score=" + str(r["score"]) + " metadata=" + str(r["metadata"]))
    if result["failed_shards"]:
        lines.append("Warning - some shards did not respond: " + str(result["failed_shards"]))

    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
