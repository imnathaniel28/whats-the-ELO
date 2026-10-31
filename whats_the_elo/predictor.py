"""Model loading and the end-to-end prediction entry point.

The network is a CNN over the board plus a small dense branch over the Stockfish
features, joined and fed to a single-layer LSTM. The exported ONNX graph is one
timestep wide, so inference walks the game ply by ply carrying (h, c) forward —
the same loop the original torch CLI ran. The prediction after the final ply is
the one that counts; the earlier ones form the trajectory the UI can plot.

Batch is fixed at 2 (white, black). That is baked into the graph and into the
original model's forward pass, so games cannot be batched together on that axis;
concurrency comes from handling requests in parallel.
"""
import threading
import time

import numpy as np

from . import elo_math
from .config import get_settings
from .engine import analysis_engine, search_limit
from .features import extract_features, game_fens, parse_pgn

LSTM_LAYERS = 1
LSTM_HIDDEN_SIZE = 64
BATCH = 2  # (white, black)

_session = None
_session_lock = threading.Lock()


def get_session():
    """Load the ONNX session once per process.

    onnxruntime sessions are safe to call concurrently from multiple threads, so
    one shared session serves every request.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import onnxruntime as ort

                settings = get_settings()
                if not settings.model_path.is_file():
                    raise FileNotFoundError(f"model not found at {settings.model_path}")

                options = ort.SessionOptions()
                options.intra_op_num_threads = settings.onnx_threads
                options.inter_op_num_threads = settings.onnx_threads
                _session = ort.InferenceSession(
                    str(settings.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
    return _session


def reset_session():
    """Drop the cached session (tests, or a model swap at runtime)."""
    global _session
    _session = None


def run_model(positions, analysis, keep_trajectory=False):
    """Step the LSTM across the game and return the final class probabilities.

    `positions` is [2, seq, 2, 8, 8] and `analysis` is [2, seq, 17].
    Returns (probabilities [2, 39], trajectory) where trajectory is a list of
    [2, 39] arrays — one per ply — or None.
    """
    session = get_session()

    if positions.ndim != 5 or positions.shape[0] != BATCH:
        raise ValueError(f"positions must be [2, seq, 2, 8, 8], got {positions.shape}")
    if analysis.ndim != 3 or analysis.shape[0] != BATCH:
        raise ValueError(f"analysis must be [2, seq, 17], got {analysis.shape}")

    hidden = np.zeros((LSTM_LAYERS, BATCH, LSTM_HIDDEN_SIZE), dtype=np.float32)
    cell = np.zeros((LSTM_LAYERS, BATCH, LSTM_HIDDEN_SIZE), dtype=np.float32)

    probabilities = None
    trajectory = [] if keep_trajectory else None

    for ply in range(positions.shape[1]):
        probabilities, hidden, cell = session.run(
            ["probs", "hn", "cn"],
            {
                "position": np.ascontiguousarray(positions[:, ply : ply + 1]),
                "evaluation": np.ascontiguousarray(analysis[:, ply : ply + 1]),
                "h0": hidden,
                "c0": cell,
            },
        )
        if keep_trajectory:
            trajectory.append(probabilities.copy())

    if probabilities is None:
        raise ValueError("game produced no plies to score")
    return probabilities, trajectory


def predict(
    pgn_text,
    scale="lichess",
    time_limit=None,
    node_limit=None,
    max_plies=None,
    include_trajectory=False,
    include_probabilities=False,
    include_fens=False,
    include_best_moves=False,
    progress=None,
):
    """Predict both players' Elo from a single game's PGN.

    This is the one function an embedding application needs. It is synchronous
    and engine-bound: expect roughly `2 * plies * time_limit` seconds.
    """
    settings = get_settings()
    game = parse_pgn(pgn_text)

    total_plies = sum(1 for _ in game.mainline_moves())
    ply_cap = settings.max_plies if max_plies is None else min(max_plies, settings.max_plies)

    started = time.monotonic()
    limit = search_limit(time_limit=time_limit, node_limit=node_limit)
    with analysis_engine() as engine:
        features = extract_features(
            game, engine, limit, max_plies=ply_cap, progress=progress
        )
    analysis_seconds = time.monotonic() - started

    probabilities, trajectory = run_model(
        features["positions"], features["analysis"], keep_trajectory=include_trajectory
    )

    raw = [elo_math.elo_from_probabilities(probabilities[side]) for side in range(BATCH)]
    scaled = [elo_math.apply_scale(value, scale) for value in raw]

    result = {
        "white_elo": elo_math.round_elo(scaled[0]),
        "black_elo": elo_math.round_elo(scaled[1]),
        "white_elo_raw": round(scaled[0], 1),
        "black_elo_raw": round(scaled[1], 1),
        "scale": scale,
        "num_moves": total_plies,
        "plies_analyzed": features["plies_analyzed"],
        "truncated": features["plies_analyzed"] < total_plies,
        "headers": features["headers"],
        "analysis_seconds": round(analysis_seconds, 3),
    }

    if include_probabilities:
        result["rating_bins"] = elo_math.RATING_BINS
        result["white_probabilities"] = probabilities[0].astype(float).tolist()
        result["black_probabilities"] = probabilities[1].astype(float).tolist()

    if include_trajectory:
        # Per-ply running Elo for each side: what the original visualizer drew.
        result["white_elo_trajectory"] = [
            elo_math.round_elo(elo_math.apply_scale(elo_math.elo_from_probabilities(step[0]), scale))
            for step in trajectory
        ]
        result["black_elo_trajectory"] = [
            elo_math.round_elo(elo_math.apply_scale(elo_math.elo_from_probabilities(step[1]), scale))
            for step in trajectory
        ]

    if include_fens:
        result["fens"] = game_fens(game, max_plies=ply_cap)
    if include_best_moves:
        result["best_moves"] = features["best_moves"]

    return result


def validate_pgn(pgn_text):
    """Cheap PGN check with no engine work, for client-side form feedback."""
    try:
        game = parse_pgn(pgn_text)
    except Exception as exc:
        return {"valid": False, "error": str(exc)}

    headers = dict(game.headers)
    num_moves = sum(1 for _ in game.mainline_moves())
    return {
        "valid": True,
        "white": headers.get("White", "Unknown"),
        "black": headers.get("Black", "Unknown"),
        "result": headers.get("Result", "*"),
        "white_elo_header": headers.get("WhiteElo"),
        "black_elo_header": headers.get("BlackElo"),
        "num_moves": num_moves,
        "estimated_seconds": round(2 * num_moves * get_settings().time_limit, 1),
    }
