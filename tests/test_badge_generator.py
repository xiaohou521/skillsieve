from clawhub_integration.badge_generator import generate_badge


def test_safe_badge():
    svg = generate_badge("safe", 0.95, 1)
    assert "<svg" in svg
    assert "SAFE" in svg
    assert "#4c1" in svg  # green


def test_malicious_badge():
    svg = generate_badge("malicious", 0.92, 3)
    assert "MALICIOUS" in svg
    assert "#e05d44" in svg  # red
    assert "L3" in svg


def test_suspicious_badge():
    svg = generate_badge("suspicious", 0.6, 2)
    assert "SUSPICIOUS" in svg
    assert "#fe7d37" in svg  # orange


def test_badge_api_endpoint():
    from fastapi.testclient import TestClient
    from clawhub_integration.api import app
    client = TestClient(app)
    response = client.get("/api/v1/badge/some-skill")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml; charset=utf-8"
    assert "<svg" in response.text
