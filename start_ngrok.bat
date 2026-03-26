@echo off
REM ============================================================
REM  OncoContext MCP Server — ngrok tunnel launcher
REM  Starts the SSE server on port 8000, then opens an ngrok tunnel.
REM  Run this script on the HOST machine.
REM ============================================================

echo.
echo ============================================================
echo   OncoContext MCP Server  ^|  ngrok Tunnel Launcher
echo ============================================================
echo.

REM ── 1. Activate virtual environment (if present) ─────────────
IF EXIST ".venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) ELSE (
    echo [1/3] No .venv found — using system Python.
)
echo.

REM ── 2. Start the SSE server in a separate window ─────────────
echo [2/3] Starting OncoContext SSE server on port 8000...
echo       (A new window titled "OncoContext SSE Server" will open)
start "OncoContext SSE Server" cmd /k python start_sse_server.py --host 0.0.0.0 --port 8000

REM Give the server a moment to start before opening ngrok
timeout /t 3 /nobreak >nul

REM ── 3. Start ngrok tunnel ────────────────────────────────────
echo [3/3] Opening ngrok tunnel to localhost:8000...
echo       Check the ngrok window or https://dashboard.ngrok.com
echo       for the public HTTPS URL (e.g. https://abc123.ngrok-free.app)
echo.
echo ============================================================
echo   NEXT STEP (remote computer):
echo     Copy remote_claude_config.json to the remote machine,
echo     replace NGROK_URL_HERE with the ngrok HTTPS URL,
echo     save to %%APPDATA%%\Claude\claude_desktop_config.json,
echo     then restart Claude Desktop.
echo ============================================================
echo.

ngrok http 8000

echo.
echo [Done] ngrok has exited. The SSE server window may still be running.
pause
