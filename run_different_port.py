"""
Run server on a different port if 8000 is busy
"""
import uvicorn
import sys

# Try to find an available port
def find_free_port(start_port=8001, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    port = 8000
    
    # Check if port 8000 is available, if not find another
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 8000))
    except OSError:
        print("⚠️  Port 8000 is already in use!")
        port = find_free_port()
        if port:
            print(f"✅ Using port {port} instead")
        else:
            print("❌ Could not find an available port")
            sys.exit(1)
    
    print("=" * 60)
    print("PerfectExam - 100% Accuracy Generator")
    print("=" * 60)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print(f"API documentation: http://localhost:{port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

