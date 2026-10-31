"""Runtime configuration, all from environment variables.

Nothing here reads a config file: a container gets its settings from the
environment, and the defaults are chosen so that `uvicorn whats_the_elo.api:app`
works with no configuration at all as long as Stockfish is on PATH.
"""
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "model.onnx"


def _env_str(name, default):
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_bool(name, default):
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def discover_engine():
    """Locate a Stockfish binary: STOCKFISH_PATH, then PATH.

    Returns None rather than raising so that /healthz can report the problem
    instead of the process failing to start.
    """
    explicit = os.environ.get("STOCKFISH_PATH", "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


@dataclass
class Settings:
    # --- model ---
    model_path: Path = field(
        default_factory=lambda: Path(_env_str("ELO_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    )
    onnx_threads: int = field(default_factory=lambda: _env_int("ELO_ONNX_THREADS", 1))

    # --- engine ---
    engine_path: "str | None" = field(default_factory=discover_engine)
    engine_threads: int = field(default_factory=lambda: _env_int("ELO_ENGINE_THREADS", 1))
    engine_hash_mb: int = field(default_factory=lambda: _env_int("ELO_ENGINE_HASH_MB", 16))

    # Search budget per position. The model was trained against `time=0.1`, so
    # that stays the default. A node limit makes results reproducible (wall-clock
    # limits make Stockfish non-deterministic) and is preferred for anything that
    # needs stable output; set ELO_NODE_LIMIT to enable it.
    time_limit: float = field(default_factory=lambda: _env_float("ELO_TIME_LIMIT", 0.1))
    node_limit: "int | None" = field(default_factory=lambda: _env_int("ELO_NODE_LIMIT", 0) or None)

    # --- request limits ---
    # Two engine calls per ply, so wall-clock is roughly 2 * plies * time_limit.
    max_plies: int = field(default_factory=lambda: _env_int("ELO_MAX_PLIES", 300))
    max_pgn_bytes: int = field(default_factory=lambda: _env_int("ELO_MAX_PGN_BYTES", 256 * 1024))
    max_concurrent: int = field(
        default_factory=lambda: _env_int("ELO_MAX_CONCURRENT", max(1, (os.cpu_count() or 2) // 2))
    )
    analysis_timeout: float = field(default_factory=lambda: _env_float("ELO_ANALYSIS_TIMEOUT", 300.0))

    # --- service ---
    api_key: "str | None" = field(default_factory=lambda: _env_str("ELO_API_KEY", "") or None)
    cors_origins: list = field(default_factory=lambda: _env_list("ELO_CORS_ORIGINS", []))
    docs_enabled: bool = field(default_factory=lambda: _env_bool("ELO_DOCS", False))

    def validate(self):
        if self.time_limit <= 0 and not self.node_limit:
            raise ValueError("ELO_TIME_LIMIT must be > 0 unless ELO_NODE_LIMIT is set")
        if self.max_plies < 2:
            raise ValueError("ELO_MAX_PLIES must be >= 2")
        if self.max_concurrent < 1:
            raise ValueError("ELO_MAX_CONCURRENT must be >= 1")
        return self


_settings = None


def get_settings():
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings().validate()
    return _settings


def reset_settings():
    """Drop the cached settings so the next call re-reads the environment (tests)."""
    global _settings
    _settings = None
