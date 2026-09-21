import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(r"D:\fake-news-bda")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

sys.stdout.reconfigure(encoding="utf-8")

client = TestClient(app)

print("=" * 70)
print("🧪 [BACKEND TEST SUITE] Starting 100% Verification of FastAPI Service")
print("=" * 70)

# -------------------------------------------------------------
# TEST 1: Root Route
# -------------------------------------------------------------
print("\n[TEST 1] Testing Root Route (GET /)...")
res = client.get("/")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
data = res.json()
assert "message" in data, "Missing 'message' in root response"
print(f" -> PASS: {data}")

# -------------------------------------------------------------
# TEST 2: Health Check Route
# -------------------------------------------------------------
print("\n[TEST 2] Testing Health Endpoint (GET /api/health)...")
res = client.get("/api/health")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
data = res.json()
assert data["status"] == "healthy", f"Status not healthy: {data}"
assert data["model_loaded"] is True, "Model not loaded"
assert data["dataset_rows"] == 67587, f"Wrong dataset rows: {data['dataset_rows']}"
print(f" -> PASS: Health check verified (Status={data['status']}, Rows={data['dataset_rows']})")

# -------------------------------------------------------------
# TEST 3: Sample Articles Route
# -------------------------------------------------------------
print("\n[TEST 3] Testing Curated Samples Endpoint (GET /api/samples)...")
res = client.get("/api/samples")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
data = res.json()
assert data["status"] == "success", "Failed to retrieve samples"
assert data["count"] >= 4, f"Expected >=4 samples, got {data['count']}"
for s in data["samples"]:
    assert "title" in s and "content" in s and "expected_label" in s
print(f" -> PASS: Retrieved {data['count']} sample articles successfully")

# -------------------------------------------------------------
# TEST 4: Validation / Bad Input Handling
# -------------------------------------------------------------
print("\n[TEST 4] Testing Request Validation (POST /api/verify with bad payload)...")
# Empty body
res = client.post("/api/verify", json={})
assert res.status_code == 422, f"Expected 422 validation error, got {res.status_code}"
# Body too short
res = client.post("/api/verify", json={"title": "a", "content": "b"})
assert res.status_code == 422, f"Expected 422 validation error, got {res.status_code}"
print(" -> PASS: Pydantic correctly rejected invalid / empty requests with HTTP 422")

# -------------------------------------------------------------
# TEST 5: Live Verification Pipeline via API
# -------------------------------------------------------------
print("\n[TEST 5] Testing Live Verification Pipeline (POST /api/verify)...")
test_payload = {
    "article_id": 34373,
    "title": "वेस्टइंडीज दौरे पर कोरोना संक्रमित हुए पाकिस्तान के मुख्य कोच मिस्बाह-उल-हक",
    "content": "पाकिस्तान क्रिकेट टीम के मुख्य कोच मिस्बाह-उल-हक वेस्टइंडीज दौरे के समापन पर कोरोना वायरस पॉजिटिव पाए गए हैं। पाकिस्तान क्रिकेट बोर्ड ने इसकी पुष्टि की है।",
    "url": "https://example.com/sports"
}

start_t = time.time()
res = client.post("/api/verify", json=test_payload)
dur = round(time.time() - start_t, 2)

assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
data = res.json()

assert data["status"] == "success", "Response status is not success"
assert data["verdict"] in ["REAL", "FAKE", "UNCERTAIN"], f"Invalid verdict: {data['verdict']}"
assert 0.0 <= data["confidence"] <= 1.0, f"Confidence out of bounds: {data['confidence']}"
assert len(data["claim"]) > 0, "Claim is empty"
assert isinstance(data["evidence"], list), "Evidence is not a list"

print(f" -> PASS: Live API call returned HTTP 200 in {dur}s")
print(f"    - Verdict: {data['verdict']} (Confidence: {data['confidence']:.2f})")
print(f"    - Extracted Claim: {data['claim']}")
print(f"    - Entities: {data['entities']}")
print(f"    - Retrieved Evidence Count: {len(data['evidence'])}")
if data["evidence"]:
    top_ev = data["evidence"][0]
    print(f"    - Top Evidence: [{top_ev.get('rank')}] {top_ev.get('title')[:80]} (Score: {top_ev.get('evidence_score'):.3f})")
print(f"    - Explanation: {data['explanation'][:120]}...")

print("\n" + "=" * 70)
print("🎉 ALL 5 BACKEND TESTS PASSED! SERVICE IS 100% OPERATIONAL")
print("=" * 70)
