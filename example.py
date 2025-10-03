import asyncio
from util.mcp import MCPClient

async def main():

    sql_query = {"sql": "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('information_schema', 'pg_catalog');"}

    client = MCPClient()

    try:
        await client.connect_to_server("/app/mcp/postgres.py")
        out = await client.call_tool("sql", sql_query)
        print(out)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())