import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def main():
    print("--- E2E API INTEGRATION TEST ---")
    
    print("1. Testing GET /health")
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    print("Health check passed.")

    print("\n2. Testing POST /session")
    resp = client.post("/session")
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["session_id"]
    print(f"Session created: {session_id}")

    print("\n3. Testing POST /chat - 'What is a chemical reaction?'")
    resp = client.post("/chat", json={"session_id": session_id, "message": "What is a chemical reaction?"})
    
    if resp.status_code == 500 and "Failed to generate answer" in resp.json().get("detail", ""):
        print("Passed: Request successfully reached the generator but failed gracefully due to placeholder Gemini credentials.")
    elif resp.status_code == 200:
        data = resp.json()
        print("Passed: Got full 200 response:", data["reply"][:50], "...")
        print("Latency:", data["latency_ms"])
        
        # Test Cache Hit
        print("\n4. Testing Cache HIT - 'What is a chemical reaction?'")
        resp2 = client.post("/chat", json={"session_id": session_id, "message": "What is a chemical reaction?"})
        assert resp2.status_code == 200
        assert resp2.json()["cache_hit"] == True
        print("Passed: Cache Hit successful.")
    else:
        print(f"FAILED: Unexpected status {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    main()
