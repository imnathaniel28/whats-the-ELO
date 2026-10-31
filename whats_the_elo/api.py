"""JSON-only HTTP service.

No templates, no static files, no HTML: this exists to be called by the chess
site/app's own backend or frontend, and every response is JSON.

    POST /v1/predict   PGN in, both players' Elo out
    POST /v1/validate  cheap PGN check, no engine work
    GET  /v1/info      model metadata and effective limits
    GET  /healthz      liveness + whether the model and engine are usable

Endpoints are declared with `def`, not `async def`, on purpose. Analysis blocks
on a Stockfish subprocess for seconds at a time; Starlette runs sync endpoints in
a worker thread, which keeps the event loop responsive. `engine.analysis_engine`
caps how many of those can run at once.
"""
import logging

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import elo_math
from .config import get_settings
from .engine import EngineBusy, EngineUnavailable, engine_version
from .features import FeatureError
from .predictor import get_session, predict, validate_pgn

log = logging.getLogger("whats_the_elo")

settings = get_settings()

app = FastAPI(
    title="What's the ELO",
    version="1.0.0",
    description="Predicts both players' Elo from a single chess game.",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


def require_api_key(x_api_key: str = Header(default="")):
    """Shared-secret check, enabled by setting ELO_API_KEY."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class PredictRequest(BaseModel):
    pgn: str = Field(..., description="The game in PGN. Only the first game is scored.")
    scale: str = Field(
        "lichess",
        description="Rating scale for the result: 'lichess' (the training scale) or 'chesscom'.",
    )
    time_limit: "float | None" = Field(
        None, ge=0.01, le=2.0, description="Seconds of Stockfish per position. Default 0.1."
    )
    node_limit: "int | None" = Field(
        None, ge=1000, description="Nodes per position instead of a time limit. Reproducible."
    )
    max_plies: "int | None" = Field(
        None, ge=2, description="Stop after this many plies."
    )
    include_probabilities: bool = Field(
        False, description="Return the full 39-bin distribution per player."
    )
    include_trajectory: bool = Field(
        False, description="Return the running Elo estimate after every ply."
    )
    include_fens: bool = Field(False, description="Return the FEN after every ply.")
    include_best_moves: bool = Field(
        False, description="Return Stockfish's best move in each position."
    )


class ValidateRequest(BaseModel):
    pgn: str


@app.exception_handler(FeatureError)
def _handle_feature_error(_request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.post("/v1/predict", dependencies=[Depends(require_api_key)])
def predict_endpoint(request: PredictRequest):
    """Score one game. Blocks for roughly 2 * plies * time_limit seconds."""
    if request.scale not in elo_math.SCALES:
        raise HTTPException(
            status_code=422, detail=f"scale must be one of {list(elo_math.SCALES)}"
        )
    if len(request.pgn.encode("utf-8")) > settings.max_pgn_bytes:
        raise HTTPException(
            status_code=413, detail=f"PGN exceeds {settings.max_pgn_bytes} bytes"
        )

    try:
        return predict(
            request.pgn,
            scale=request.scale,
            time_limit=request.time_limit,
            node_limit=request.node_limit,
            max_plies=request.max_plies,
            include_probabilities=request.include_probabilities,
            include_trajectory=request.include_trajectory,
            include_fens=request.include_fens,
            include_best_moves=request.include_best_moves,
        )
    except FeatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EngineBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EngineUnavailable as exc:
        log.error("engine unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc


@app.post("/v1/validate", dependencies=[Depends(require_api_key)])
def validate_endpoint(request: ValidateRequest):
    """Check a PGN parses and report move count and headers. No engine work."""
    if len(request.pgn.encode("utf-8")) > settings.max_pgn_bytes:
        raise HTTPException(
            status_code=413, detail=f"PGN exceeds {settings.max_pgn_bytes} bytes"
        )
    return validate_pgn(request.pgn)


@app.get("/v1/info")
def info_endpoint():
    """Model metadata and the limits this instance is running with."""
    return {
        "model": {
            "path": str(settings.model_path),
            "num_classes": elo_math.NUM_CLASSES,
            "rating_bins": elo_math.RATING_BINS,
            "trained_on": "lichess database, June 2018, ~20k games",
            "output": "probability per 100-point rating band; Elo is the weighted mean of band midpoints",
        },
        "accuracy": {
            "within_200": 0.605,
            "within_400": 0.877,
            "within_600": 0.966,
            "note": "Measured on 3000 held-out games. Overestimates below 1000 and underestimates above 2600.",
        },
        "limits": {
            "max_plies": settings.max_plies,
            "max_pgn_bytes": settings.max_pgn_bytes,
            "max_concurrent": settings.max_concurrent,
            "default_time_limit": settings.time_limit,
            "node_limit": settings.node_limit,
        },
        "scales": list(elo_math.SCALES),
    }


@app.get("/healthz")
def healthz():
    """Liveness and readiness. 503 when the model or engine is unusable."""
    model_ok, model_error = True, None
    try:
        get_session()
    except Exception as exc:
        model_ok, model_error = False, str(exc)

    engine_ok = settings.engine_path is not None
    body = {
        "status": "ok" if (model_ok and engine_ok) else "degraded",
        "model_loaded": model_ok,
        "model_error": model_error,
        "engine_found": engine_ok,
        "engine_path": settings.engine_path,
    }
    if not (model_ok and engine_ok):
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/healthz/engine")
def healthz_engine():
    """Actually start Stockfish and report its version. Slower than /healthz."""
    name = engine_version()
    if name is None:
        raise HTTPException(
            status_code=503,
            detail={"engine_found": False, "engine_path": settings.engine_path},
        )
    return {"engine_found": True, "engine": name, "engine_path": settings.engine_path}
