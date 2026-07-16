from pathlib import Path
from fastapi.testclient import TestClient
from clawhub_integration.api import app

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_scan_benign():
    response = client.post("/api/v1/scan", json={
        "skill_path": str(FIXTURES / "benign_skill"),
        "max_layer": 1,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "safe"
    assert data["report_id"] is None

def test_scan_not_found():
    response = client.post("/api/v1/scan", json={
        "skill_path": "/nonexistent/path",
        "max_layer": 1,
    })
    assert response.status_code == 404

def test_report_not_found():
    response = client.get("/api/v1/report/nonexistent")
    assert response.status_code == 404

def test_webhook_skill_submitted():
    response = client.post("/api/v1/webhook/clawhub", json={
        "event_type": "skill_submitted",
        "skill_name": "test-skill",
        "skill_url": "https://clawhub.ai/user/test-skill",
        "author": "testuser",
        "timestamp": "2026-04-03T12:00:00Z",
    })
    assert response.status_code == 200
    assert response.json()["accepted"] is True

def test_webhook_unknown_event():
    response = client.post("/api/v1/webhook/clawhub", json={
        "event_type": "unknown",
        "skill_name": "test",
        "skill_url": "https://clawhub.ai/x",
        "author": "x",
        "timestamp": "2026-04-03T12:00:00Z",
    })
    assert response.status_code == 400
