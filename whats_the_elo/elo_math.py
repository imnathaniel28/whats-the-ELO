"""Rating-range bins and the decoding of model output into an Elo number.

Numpy port of the inference path of the original `elo_ai/helper_functions/elo_range.py`.
The model does not regress an Elo directly: it emits a probability over 39
100-point rating bands, and the reported Elo is the probability-weighted mean of
the band midpoints.
"""
import numpy as np

RANGE_START = 0
RANGE_END = 4000
RANGE_STEP = 100

# (low, high) for each band: (0,100), (100,200) ... (3800,3900). 39 bands.
BIN_LOWS = np.arange(RANGE_START, RANGE_END - RANGE_STEP, RANGE_STEP, dtype=np.float32)
BIN_HIGHS = np.arange(RANGE_START + RANGE_STEP, RANGE_END, RANGE_STEP, dtype=np.float32)
BIN_MIDPOINTS = (BIN_LOWS + BIN_HIGHS) / 2.0
NUM_CLASSES = len(BIN_MIDPOINTS)

RATING_BINS = [(int(low), int(high)) for low, high in zip(BIN_LOWS, BIN_HIGHS)]


def elo_from_probabilities(probabilities):
    """Probability-weighted mean of the rating-band midpoints.

    `probabilities` is a 1-D array of NUM_CLASSES values for one player.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probabilities.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"expected {NUM_CLASSES} class probabilities, got {probabilities.shape[0]}"
        )
    return float(np.sum(probabilities * BIN_MIDPOINTS))


def round_elo(elo):
    """Round to the nearest 50, as the original CLI reported it."""
    return int(50 * round(float(elo) / 50))


def to_chesscom(elo):
    """Convert a lichess-scale prediction to the chess.com scale.

    The model is trained on lichess games. Blitz conversion formula from the
    lichess forum discussion referenced by the original project; chess.com
    ratings run roughly 400 points lower.
    """
    return 1.138 * float(elo) - 665


SCALES = ("lichess", "chesscom")


def apply_scale(elo, scale):
    if scale == "lichess":
        return float(elo)
    if scale == "chesscom":
        return to_chesscom(elo)
    raise ValueError(f"unknown rating scale {scale!r}; expected one of {SCALES}")
