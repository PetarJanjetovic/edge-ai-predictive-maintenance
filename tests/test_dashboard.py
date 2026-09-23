import time

from fastapi.testclient import TestClient

from edge_pdm.dashboard import create_app


def test_dashboard_accepts_and_returns_telemetry():
    client = TestClient(create_app())
    payload = {
        "device_id": "test-device",
        "timestamp": time.time(),
        "prediction": "normal",
        "confidence": 0.95,
        "probabilities": {"normal": 0.95},
        "rms_g": 0.12,
        "inference_ms": 2.3,
    }
    assert client.post("/api/telemetry", json=payload).status_code == 202
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.json()[0]["device_id"] == "test-device"


def test_dashboard_home_and_health():
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok", "samples": 0}

