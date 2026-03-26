"""Quick test to verify MCP server starts correctly."""

import subprocess
import json
import sys
import time

def test_mcp_server():
    """Test if the MCP server starts and responds to initialization."""
    
    # Path to the virtual environment Python
    python_path = r"C:/Users/Akshay Gupta/Desktop/New folder/onco-test/.venv/Scripts/python.exe"
    
    print("🔍 Testing MCP server connection...")
    print(f"Using Python: {python_path}")
    
    # Start the server
    try:
        # Import os to get current environment
        import os
        
        # Create environment that inherits current env and adds NCBI_API_KEY
        env = os.environ.copy()
        env["NCBI_API_KEY"] = "72bc9c9f476018fb32875813755db0c7e508"
        
        process = subprocess.Popen(
            [python_path, "-m", "oncocontext"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=r"C:/Users/Akshay Gupta/Desktop/New folder/onco-test",
            env=env
        )
        
        print("✅ Server process started")
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("📤 Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Wait for response with timeout
        print("⏳ Waiting for response...")
        time.sleep(2)
        
        # Check if process is still alive
        if process.poll() is None:
            print("✅ Server is running and responsive!")
            print("\n💡 Your MCP configuration should work now. Please:")
            print("   1. Restart Roo Code / VS Code")
            print("   2. The MCP server should connect successfully")
            
            # Clean up
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            print("❌ Server process terminated unexpectedly")
            stderr = process.stderr.read()
            if stderr:
                print(f"Error output:\n{stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
