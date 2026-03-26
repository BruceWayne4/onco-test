"""Interactive MCP client for testing the OncoContext MCP server."""

import subprocess
import json
import sys
import os
import time
from typing import Optional

class MCPClient:
    """Simple MCP client for interactive testing."""
    
    def __init__(self, python_path: str, cwd: str):
        self.python_path = python_path
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    def start_server(self):
        """Start the MCP server process."""
        print("🚀 Starting MCP server...")
        
        # Inherit current environment and add NCBI_API_KEY
        env = os.environ.copy()
        env["NCBI_API_KEY"] = "72bc9c9f476018fb32875813755db0c7e508"
        
        try:
            self.process = subprocess.Popen(
                [self.python_path, "-m", "oncocontext"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env
            )
            print("✅ Server started successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request to the server."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        try:
            request_json = json.dumps(request) + "\n"
            if self.process and self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()
                
                # Read response
                if self.process.stdout:
                    response_line = self.process.stdout.readline()
                    if response_line:
                        return json.loads(response_line)
            
            return {"error": "No response from server"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def initialize(self):
        """Initialize the MCP connection."""
        print("📤 Initializing connection...")
        response = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "interactive-test-client",
                "version": "1.0.0"
            }
        })
        
        if "result" in response:
            print("✅ Connection initialized!")
            print(f"   Server: {response['result'].get('serverInfo', {}).get('name', 'Unknown')}")
            print(f"   Version: {response['result'].get('serverInfo', {}).get('version', 'Unknown')}\n")
            return True
        else:
            print(f"❌ Initialization failed: {response.get('error', 'Unknown error')}\n")
            return False
    
    def list_tools(self):
        """List all available tools."""
        print("📋 Listing available tools...")
        response = self.send_request("tools/list")
        
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"\n✅ Found {len(tools)} tools:\n")
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool['name']}")
                print(f"   {tool.get('description', 'No description')[:80]}...")
                print()
            return tools
        else:
            print(f"❌ Failed to list tools: {response.get('error', 'Unknown error')}\n")
            return []
    
    def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific tool with arguments."""
        print(f"🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {json.dumps(arguments, indent=2)}\n")
        
        response = self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if "result" in response:
            print("✅ Tool executed successfully!\n")
            result = response["result"]
            
            # Pretty print the result
            if isinstance(result, dict):
                print("📊 Result:")
                print(json.dumps(result, indent=2))
            else:
                print(f"📊 Result: {result}")
            print()
            return result
        else:
            print(f"❌ Tool execution failed: {response.get('error', 'Unknown error')}\n")
            return None
    
    def shutdown(self):
        """Shutdown the server."""
        if self.process:
            print("\n🛑 Shutting down server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("✅ Server stopped successfully")
            except Exception as e:
                print(f"⚠️  Warning during shutdown: {e}")
                self.process.kill()

def interactive_mode(client: MCPClient):
    """Run interactive mode."""
    tools = []
    
    print("\n" + "="*60)
    print("🧪 INTERACTIVE MCP TEST CLIENT")
    print("="*60)
    print("\nCommands:")
    print("  list    - List all available tools")
    print("  search  - Quick search literature")
    print("  custom  - Call any tool with custom arguments")
    print("  quit    - Exit\n")
    
    while True:
        try:
            command = input("mcp> ").strip().lower()
            
            if command == "quit" or command == "exit":
                break
            
            elif command == "list":
                tools = client.list_tools()
            
            elif command == "search":
                query = input("Enter search query: ").strip()
                if query:
                    client.call_tool("tool_search_literature", {
                        "query": query,
                        "max_results": 5
                    })
            
            elif command == "custom":
                if not tools:
                    tools = client.list_tools()
                
                print("\nAvailable tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool['name']}")
                
                try:
                    choice = int(input("\nSelect tool number: "))
                    if 1 <= choice <= len(tools):
                        tool = tools[choice - 1]
                        print(f"\nTool: {tool['name']}")
                        print(f"Parameters: {json.dumps(tool.get('inputSchema', {}), indent=2)}\n")
                        
                        args_str = input("Enter arguments as JSON: ").strip()
                        args = json.loads(args_str) if args_str else {}
                        
                        client.call_tool(tool['name'], args)
                    else:
                        print("Invalid choice")
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Error: {e}")
            
            elif command:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            break

def main():
    """Main entry point."""
    python_path = r"C:/Users/Akshay Gupta/Desktop/New folder/onco-test/.venv/Scripts/python.exe"
    cwd = r"C:/Users/Akshay Gupta/Desktop/New folder/onco-test"
    
    client = MCPClient(python_path, cwd)
    
    try:
        # Start server
        if not client.start_server():
            return 1
        
        # Give server time to start
        time.sleep(1)
        
        # Initialize connection
        if not client.initialize():
            return 1
        
        # List tools initially
        client.list_tools()
        
        # Enter interactive mode
        interactive_mode(client)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        client.shutdown()

if __name__ == "__main__":
    sys.exit(main())
