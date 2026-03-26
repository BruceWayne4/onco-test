"""Start the OncoContext MCP server with SSE (Server-Sent Events) transport.

This script runs the server over HTTP so ngrok (or any HTTP tunnel) can expose
it to remote Claude Desktop clients.

FastMCP.run(transport="sse") reads host/port from mcp.settings, so we patch
those before calling run().

Usage:
    python start_sse_server.py [--host HOST] [--port PORT]

Defaults:
    host = 0.0.0.0   (listen on all interfaces, required for ngrok)
    port = 8000
"""

import argparse
import sys
from pathlib import Path

# ── ensure the project src/ is on the path when run directly ──────────────────
_repo_root = Path(__file__).parent
sys.path.insert(0, str(_repo_root / "src"))

from oncocontext.server import mcp  # noqa: E402  (import after path fixup)
from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OncoContext MCP server via SSE transport")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    # Patch the settings that FastMCP.run_sse_async() reads from
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # ── Disable DNS rebinding protection for tunnel/ngrok usage ───────────────
    # FastMCP auto-enables DNS rebinding protection when the initial host is
    # "127.0.0.1" / "localhost" / "::1".  This rejects any request whose Host
    # header does not match localhost (HTTP 421 "Misdirected Request").
    # When we serve behind ngrok the Host header is the ngrok domain, so we must
    # disable the check — otherwise every /sse request fails with 421.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    print(f"[OncoContext] Starting MCP SSE server on http://{args.host}:{args.port}")
    print(f"[OncoContext] SSE endpoint  : http://{args.host}:{args.port}/sse")
    print(f"[OncoContext] Health check  : http://localhost:{args.port}/")
    print("[OncoContext] DNS rebinding protection: DISABLED (ngrok/tunnel mode)")
    print("[OncoContext] Press Ctrl+C to stop.")
    print()

    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
