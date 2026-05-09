import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, max_concurrent: int = 0):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Optional semaphore to limit concurrent call_tool invocations
        # (protects Postgres from too many concurrent queries).
        # 0 = no limit (default, preserves existing behavior).
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    async def cleanup(self):
        """Clean up the MCP client connection.

        The MCP stdio_client uses anyio cancel scopes internally which are
        task-bound. Since connect_to_server and cleanup typically run in
        different asyncio tasks (via separate run_until_complete calls),
        the normal AsyncExitStack.aclose() raises a RuntimeError.

        We catch that error here since the MCP server subprocess will be
        cleaned up automatically when the parent Python process exits.
        """
        try:
            await self.exit_stack.aclose()
        except RuntimeError as e:
            if "cancel scope" in str(e).lower() or "different task" in str(e).lower():
                # Expected when cleanup runs in a different task than connect.
                # The subprocess will be reaped on parent process exit.
                print(
                    "[MCPClient] Suppressed cancel-scope cleanup error "
                    "(MCP subprocess will exit with parent process).",
                    file=sys.stderr,
                )
            else:
                raise
        except Exception as e:
            # Catch any other unexpected cleanup errors to avoid crashing
            # after training has already completed successfully.
            print(
                f"[MCPClient] Suppressed unexpected cleanup error: {e}",
                file=sys.stderr,
            )

    async def connect_to_server(self, server_script_path: str):

        command = "python3"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

    async def list_tools(self):
        response = await self.session.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def call_tool(self, tool, tool_input, timeout: float | None = 60.0):
        """Call an MCP tool, optionally with a hard timeout.

        A stdio MCP session with no timeout can wedge the entire event loop if
        the server stops responding mid-request. Defaulting to 60s lets callers
        surface a timeout instead of hanging forever. Callers that already wrap
        the call in asyncio.wait_for should pass timeout=None.
        """
        async def _do_call():
            if timeout is None:
                return await self.session.call_tool(tool, tool_input)
            return await asyncio.wait_for(
                self.session.call_tool(tool, tool_input), timeout=timeout
            )

        if self._sem:
            async with self._sem:
                return await _do_call()
        return await _do_call()
