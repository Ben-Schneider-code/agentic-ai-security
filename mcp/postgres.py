from mcp.server.fastmcp import FastMCP
import os 
import asyncpg
import json

mcp = FastMCP("postgres")

# ENV PGHOST=localhost
# ENV PGUSER=julia
# ENV PGPASSWORD=123
# ENV PGDATABASE=msft_customers

#TODO fix to get this from env
DB_CONFIG = {
    "user": "julia",
    "password": "123",
    "database": "msft_customers",
    "host": "localhost",
    "port": 5432,
}

@mcp.tool()
async def sql(sql: str) -> str:

    """Run a SQL query against the local PostgreSQL database and return results as JSON."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            if sql.strip().lower().startswith("select"):
                rows = await conn.fetch(sql)
                # Convert each row to a dictionary
                result = [dict(row) for row in rows]
            else:
                await conn.execute(sql)
                result = {"status": "success", "message": "Query executed successfully"}
        finally:
            await conn.close()
        return str(result)
    except Exception as e:
        return str({"status": "error", "message": str(e)})

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
