"""The /v1/stats endpoint and the statistics page served at /stats."""

from fastapi.testclient import TestClient

from common import auth
from common.stats import stats as stats_ledger
from common.tabby_config import config
from endpoints.server import setup_app

API_KEY = "test-api-key"
ADMIN_KEY = "test-admin-key"

RESPONSE_KEYS = {
    "server",
    "model",
    "gpu",
    "live",
    "session",
    "lifetime",
    "models",
    "history",
    "incidents",
}


def generation(**overrides):
    base = {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "prompt_time": 0.06,
        "gen_tokens": 50,
        "gen_time": 1.2,
        "queue_time": 0.0,
        "total_time": 1.26,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
    }
    base.update(overrides)
    return base


def client_with_keys(monkeypatch):
    monkeypatch.setattr(auth, "DISABLE_AUTH", False)
    monkeypatch.setattr(
        auth, "AUTH_KEYS", auth.AuthKeys(api_key=API_KEY, admin_key=ADMIN_KEY)
    )
    return TestClient(setup_app())


def test_stats_does_not_require_a_key(monkeypatch):
    client = client_with_keys(monkeypatch)

    response = client.get("/v1/stats")
    assert response.status_code == 200
    assert set(response.json()) == RESPONSE_KEYS


def test_stats_accepts_an_api_key(monkeypatch):
    client = client_with_keys(monkeypatch)

    response = client.get("/v1/stats", headers={"Authorization": f"Bearer {API_KEY}"})

    assert response.status_code == 200
    assert set(response.json()) == RESPONSE_KEYS


def test_stats_accepts_an_admin_key(monkeypatch):
    client = client_with_keys(monkeypatch)

    response = client.get("/v1/stats", headers={"x-admin-key": ADMIN_KEY})

    assert response.status_code == 200


def test_stats_reports_recorded_activity(monkeypatch):
    client = client_with_keys(monkeypatch)
    before = stats_ledger.session["requests"]

    stats_ledger.record_generation(generation(), model="llama-8b")

    body = client.get("/v1/stats", headers={"Authorization": f"Bearer {API_KEY}"}).json()

    assert body["session"]["requests"] == before + 1
    assert body["session"]["gen_tokens"] == 50
    assert body["session"]["processed_tokens"] == 100


def test_stats_answer_without_a_model(monkeypatch):
    client = client_with_keys(monkeypatch)

    body = client.get("/v1/stats", headers={"Authorization": f"Bearer {API_KEY}"}).json()

    assert body["live"] is None
    assert body["model"] is None
    assert isinstance(body["history"]["hourly"], list)


def test_stats_page_needs_no_key(monkeypatch):
    monkeypatch.setattr(config.network, "stats_page", True)
    monkeypatch.setattr(auth, "DISABLE_AUTH", False)
    monkeypatch.setattr(auth, "AUTH_KEYS", None)
    client = TestClient(setup_app())

    response = client.get("/stats", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/stats/"

    body = client.get("/stats/")
    assert body.status_code == 200
    assert "statistics" in body.text.lower()


def test_stats_page_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config.network, "stats_page", False)
    monkeypatch.setattr(auth, "DISABLE_AUTH", False)
    monkeypatch.setattr(auth, "AUTH_KEYS", None)
    client = TestClient(setup_app())

    assert client.get("/stats", follow_redirects=False).status_code == 404