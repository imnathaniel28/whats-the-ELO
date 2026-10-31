"""HTTP contract tests. Only the /v1/predict cases need Stockfish."""
import pytest
from fastapi.testclient import TestClient

from whats_the_elo.api import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Metadata, no engine needed
# ---------------------------------------------------------------------------
def test_info_reports_bins_and_limits():
    body = client.get("/v1/info").json()
    assert body["model"]["num_classes"] == 39
    assert body["model"]["rating_bins"][0] == [0, 100]
    assert body["scales"] == ["lichess", "chesscom"]
    assert body["limits"]["max_plies"] >= 2


def test_healthz_reports_model_and_engine():
    response = client.get("/healthz")
    body = response.json() if response.status_code == 200 else response.json()["detail"]
    assert body["model_loaded"] is True
    assert "engine_found" in body


def test_no_html_is_served():
    # The service is JSON-only; there is deliberately no index page.
    assert client.get("/").status_code == 404


# ---------------------------------------------------------------------------
# Validation endpoint
# ---------------------------------------------------------------------------
def test_validate_accepts_a_real_game(pgn_text):
    body = client.post("/v1/validate", json={"pgn": pgn_text}).json()
    assert body["valid"] is True
    assert body["num_moves"] == 67
    assert body["white_elo_header"] == "1714"
    assert body["estimated_seconds"] > 0


def test_validate_reports_why_a_pgn_is_bad():
    body = client.post("/v1/validate", json={"pgn": "not a game"}).json()
    assert body["valid"] is False
    assert body["error"]


def test_validate_requires_the_pgn_field():
    assert client.post("/v1/validate", json={}).status_code == 422


def test_oversized_pgn_is_refused():
    huge = "1. e4 e5 " * 200_000
    response = client.post("/v1/validate", json={"pgn": huge})
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Request validation on /v1/predict (rejected before any engine work)
# ---------------------------------------------------------------------------
def test_unknown_scale_is_refused(pgn_text):
    response = client.post("/v1/predict", json={"pgn": pgn_text, "scale": "fide"})
    assert response.status_code == 422


def test_empty_pgn_is_refused():
    response = client.post("/v1/predict", json={"pgn": ""})
    assert response.status_code == 422


def test_out_of_band_time_limit_is_refused(pgn_text):
    response = client.post("/v1/predict", json={"pgn": pgn_text, "time_limit": 99})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Full prediction
# ---------------------------------------------------------------------------
@pytest.mark.engine
def test_predict_returns_both_ratings(short_pgn_text):
    response = client.post(
        "/v1/predict",
        json={"pgn": short_pgn_text, "time_limit": 0.01, "max_plies": 12},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert 0 <= body["white_elo"] <= 3900
    assert 0 <= body["black_elo"] <= 3900
    assert body["white_elo"] % 50 == 0
    assert body["scale"] == "lichess"
    assert body["plies_analyzed"] == 12
    assert body["truncated"] is True
    assert body["analysis_seconds"] > 0
    # Optional payloads stay off unless asked for.
    assert "white_probabilities" not in body
    assert "fens" not in body


@pytest.mark.engine
def test_predict_optional_payloads(short_pgn_text):
    response = client.post(
        "/v1/predict",
        json={
            "pgn": short_pgn_text,
            "time_limit": 0.01,
            "max_plies": 8,
            "include_probabilities": True,
            "include_trajectory": True,
            "include_fens": True,
            "include_best_moves": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["white_probabilities"]) == 39
    assert sum(body["white_probabilities"]) == pytest.approx(1.0, abs=1e-4)
    assert len(body["rating_bins"]) == 39
    assert len(body["white_elo_trajectory"]) == 4  # 8 plies -> 4 per side
    assert len(body["fens"]) == 9                  # 8 plies plus the start
    assert len(body["best_moves"]) == 8


@pytest.mark.engine
def test_chesscom_scale_is_lower_than_lichess(short_pgn_text):
    payload = {"pgn": short_pgn_text, "node_limit": 20_000, "max_plies": 10}
    lichess = client.post("/v1/predict", json={**payload, "scale": "lichess"}).json()
    chesscom = client.post("/v1/predict", json={**payload, "scale": "chesscom"}).json()
    assert chesscom["white_elo_raw"] < lichess["white_elo_raw"]


@pytest.mark.engine
def test_node_limit_makes_results_reproducible(short_pgn_text):
    # A node budget pins the search; a wall-clock budget would not.
    payload = {"pgn": short_pgn_text, "node_limit": 20_000, "max_plies": 10}
    first = client.post("/v1/predict", json=payload).json()
    second = client.post("/v1/predict", json=payload).json()
    assert first["white_elo_raw"] == second["white_elo_raw"]
    assert first["black_elo_raw"] == second["black_elo_raw"]
