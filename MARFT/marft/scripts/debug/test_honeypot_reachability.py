import sys
import asyncio
import asyncpg
from pathlib import Path

# Setup paths to import marft code
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from marft.envs.redteam_sql.redteam_sql_env import REWARD_CONFIG

DB_CONFIG = {
    "user": "agent_user",
    "password": "db_agent_password",
    "database": "msft_customers",
    "host": "localhost",
    "port": 5432,
}

# Also need an admin connection to turn rules on/off to identify the blocker
ADMIN_DB_CONFIG = {
    "user": "postgres",
    "password": "postgres123",
    "database": "msft_customers",
    "host": "localhost",
    "port": 5432,
}


async def test_query(conn, query: str) -> tuple[bool, str]:
    """Test a query and return whether it succeeded and the error message if it failed."""
    try:
        await conn.fetch(query)
        return True, ""
    except asyncpg.exceptions.InsufficientPrivilegeError as e:
        return False, f"Permission Denied: {e}"
    except Exception as e:
        return False, str(e)


async def main():
    print("=========================================================")
    print("   The 'God Mode' Degradation Test: Honeypot Reachability")
    print("=========================================================")
    print("Bypassing Blueteam LLM completely and testing DB directly...\n")

    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        admin_conn = await asyncpg.connect(**ADMIN_DB_CONFIG)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    reachable_count = 0
    total_count = 0
    impossible_honeypots = []

    # 1. Test Honeypot Tables
    print(f"--- Testing {len(REWARD_CONFIG.honeypot_tables)} Honeypot Tables ---")
    for table in REWARD_CONFIG.honeypot_tables:
        total_count += 1
        query = f"SELECT * FROM public.{table} LIMIT 1;"
        success, error = await test_query(conn, query)
        if success:
            print(f"[✓] REACHABLE: Table '{table}'")
            reachable_count += 1
        else:
            print(f"[✗] UNREACHABLE: Table '{table}' - {error}")
            impossible_honeypots.append(("table", table, query))

    # 2. Test Honeypot Columns in Customer table
    print(
        f"\n--- Testing {len(REWARD_CONFIG.honeypot_columns)} Honeypot Columns (Customer table) ---"
    )
    for col in REWARD_CONFIG.honeypot_columns:
        total_count += 1
        query = f"SELECT {col} FROM public.customer LIMIT 1;"
        success, error = await test_query(conn, query)
        if success:
            print(f"[✓] REACHABLE: Column 'customer.{col}'")
            reachable_count += 1
        else:
            print(f"[✗] UNREACHABLE: Column 'customer.{col}' - {error}")
            impossible_honeypots.append(("column", f"customer.{col}", query))

    # 3. Test Honeypot Columns in SalesOrderHeader table
    print(
        f"\n--- Testing {len(REWARD_CONFIG.honeypot_salesorder_columns)} Honeypot Columns (SalesOrderHeader) ---"
    )
    for col in REWARD_CONFIG.honeypot_salesorder_columns:
        total_count += 1
        query = f"SELECT {col} FROM public.salesorderheader LIMIT 1;"
        success, error = await test_query(conn, query)
        if success:
            print(f"[✓] REACHABLE: Column 'salesorderheader.{col}'")
            reachable_count += 1
        else:
            print(f"[✗] UNREACHABLE: Column 'salesorderheader.{col}' - {error}")
            impossible_honeypots.append(("column", f"salesorderheader.{col}", query))

    # 4. Test Honeypot Order IDs
    print(
        f"\n--- Testing {len(REWARD_CONFIG.honeypot_order_ids)} Honeypot Order IDs ---"
    )
    for oid in REWARD_CONFIG.honeypot_order_ids:
        total_count += 1
        query = f"SELECT salesorderid FROM public.salesorderheader WHERE salesorderid = {oid};"
        success, error = await test_query(conn, query)
        if success:
            print(f"[✓] REACHABLE: Order ID {oid}")
            reachable_count += 1
        else:
            print(f"[✗] UNREACHABLE: Order ID {oid} - {error}")
            impossible_honeypots.append(("order_id", str(oid), query))

    # 5. Test Honeypot GUIDs
    print(f"\n--- Testing {len(REWARD_CONFIG.honeypot_guids)} Honeypot GUIDs ---")
    for guid in REWARD_CONFIG.honeypot_guids:
        total_count += 1
        # It could be in salesorderheader or salesorderdetail. Test both.
        q1 = f"SELECT rowguid FROM public.salesorderheader WHERE rowguid = '{guid}';"
        q2 = f"SELECT rowguid FROM public.salesorderdetail WHERE rowguid = '{guid}';"
        success1, error1 = await test_query(conn, q1)
        success2, error2 = await test_query(conn, q2)

        if success1 or success2:
            print(f"[✓] REACHABLE: GUID '{guid}'")
            reachable_count += 1
        else:
            err = error1 if not success1 else error2
            print(f"[✗] UNREACHABLE: GUID '{guid}' - {err}")
            impossible_honeypots.append(("guid", guid, q1))

    print("\n=========================================================")
    print(
        f"SUMMARY: {reachable_count}/{total_count} honeypots are mathematically reachable."
    )
    print("=========================================================")

    if impossible_honeypots:
        print(
            "\nNow attempting to sequentially turn off DB defensive rules to identify blockers for unreachable honeypots...\n"
        )

        for htype, identifier, query in impossible_honeypots:
            blocker_found = False
            print(f"Analyzing unreachable honeypot: {identifier} ({htype})")

            if htype == "table":
                # Rule: GRANT SELECT
                print(
                    f"  -> Rule to test: Missing 'GRANT SELECT ON public.{identifier} TO agent_role'"
                )
                await admin_conn.execute(
                    f"GRANT SELECT ON public.{identifier} TO agent_role;"
                )
                success, error = await test_query(conn, query)
                if success:
                    print(
                        f"  -> [IDENTIFIED] Once 'GRANT SELECT ON public.{identifier}' is applied, honeypot is reachable!"
                    )
                    blocker_found = True
                await admin_conn.execute(
                    f"REVOKE SELECT ON public.{identifier} FROM agent_role;"
                )

            elif htype == "column" and "customer." in identifier:
                col = identifier.split(".")[1]
                print(
                    f"  -> Rule to test: Column-level lack of GRANT in 'GRANT SELECT (customerid, firstname, lastname) ON public.customer TO agent_role'"
                )
                await admin_conn.execute(
                    f"GRANT SELECT ({col}) ON public.customer TO agent_role;"
                )
                success, error = await test_query(conn, query)
                if success:
                    print(
                        f"  -> [IDENTIFIED] Once 'GRANT SELECT ({col}) ON public.customer' is applied, honeypot is reachable!"
                    )
                    blocker_found = True
                await admin_conn.execute(
                    f"REVOKE SELECT ({col}) ON public.customer FROM agent_role;"
                )

            # Note: For other types, they might fail due to something else or be completely missing
            if not blocker_found:
                print(
                    f"  -> [IMPOSSIBLE] Evaluating feasibility: No obvious single rule bypass makes this reachable. It might be literally impossible without a major SQL trick or tweaking the rule."
                )

    await conn.close()
    await admin_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
