from mcp.server.fastmcp import FastMCP
import os 
import asyncpg
import json

mcp = FastMCP("postgres")

# Connection details come from the AAS_DB_* contract exported by
# script/pg_ephemeral.sh (or, inside the legacy container, set explicitly by
# dockerfile.base / init-docker-compose.sh). There is intentionally NO default:
# a missing variable means the ephemeral Postgres was never provisioned, and we
# crash early here rather than silently connecting to a wrong/absent server.
#
# The agents connect as the restricted `agent_user` (AAS_DB_AGENT_USER), never
# the privileged bootstrap superuser.
def _build_db_config() -> dict:
    required = {
        "user": "AAS_DB_AGENT_USER",
        "password": "AAS_DB_AGENT_PASSWORD",
        "database": "AAS_DB_NAME",
        "host": "AAS_DB_HOST",
        "port": "AAS_DB_PORT",
    }
    missing = [env for env in required.values() if not os.environ.get(env)]
    if missing:
        raise RuntimeError(
            "mcp/postgres.py: missing DB connection env vars: "
            f"{', '.join(missing)}. The ephemeral Postgres was not provisioned "
            "— start it via script/pg_ephemeral.sh."
        )
    cfg = {key: os.environ[env] for key, env in required.items()}
    cfg["port"] = int(cfg["port"])
    return cfg


DB_CONFIG = _build_db_config()

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
