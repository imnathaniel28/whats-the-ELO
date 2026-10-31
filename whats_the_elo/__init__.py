"""What's the ELO — headless Elo prediction from a single chess game.

Embedding in a Python backend:

    from whats_the_elo import predict
    result = predict(pgn_text)
    result["white_elo"], result["black_elo"]

Embedding in anything else: run the HTTP service in `whats_the_elo.api` and
POST to /v1/predict. See README.md.
"""
from .elo_math import NUM_CLASSES, RATING_BINS, elo_from_probabilities, round_elo, to_chesscom
from .predictor import predict, validate_pgn

__version__ = "1.0.0"

__all__ = [
    "predict",
    "validate_pgn",
    "elo_from_probabilities",
    "round_elo",
    "to_chesscom",
    "RATING_BINS",
    "NUM_CLASSES",
    "__version__",
]
