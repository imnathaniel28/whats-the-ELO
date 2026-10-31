import numpy as np
import pytest

from whats_the_elo import elo_math


def test_bins_match_the_trained_head():
    # The model's final layer has 39 outputs; the bins must line up exactly.
    assert elo_math.NUM_CLASSES == 39
    assert elo_math.RATING_BINS[0] == (0, 100)
    assert elo_math.RATING_BINS[-1] == (3800, 3900)


def test_elo_is_the_probability_weighted_midpoint():
    probabilities = np.zeros(39, dtype=np.float32)
    probabilities[15] = 1.0  # the (1500, 1600) band
    assert elo_math.elo_from_probabilities(probabilities) == pytest.approx(1550.0)


def test_split_mass_averages_the_midpoints():
    probabilities = np.zeros(39, dtype=np.float32)
    probabilities[10] = 0.5  # midpoint 1050
    probabilities[20] = 0.5  # midpoint 2050
    assert elo_math.elo_from_probabilities(probabilities) == pytest.approx(1550.0)


def test_wrong_class_count_is_rejected():
    with pytest.raises(ValueError, match="39 class probabilities"):
        elo_math.elo_from_probabilities(np.zeros(10, dtype=np.float32))


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1719.2, 1700),
        (1024.0, 1000),
        (1026.0, 1050),
        (0.0, 0),
        # Exact .5 cases round half-to-even, matching the torch.round the
        # original used: 1725/50 = 34.5 -> 34 -> 1700, not 1750.
        (1725.0, 1700),
        (1775.0, 1800),
    ],
)
def test_round_elo_snaps_to_fifty(raw, expected):
    assert elo_math.round_elo(raw) == expected


def test_chesscom_conversion_lowers_the_rating():
    # The published blitz formula; chess.com runs roughly 400 points lower.
    assert elo_math.to_chesscom(1700) == pytest.approx(1.138 * 1700 - 665)
    assert elo_math.to_chesscom(1700) < 1700


def test_apply_scale_rejects_unknown_scales():
    assert elo_math.apply_scale(1500, "lichess") == 1500.0
    with pytest.raises(ValueError, match="unknown rating scale"):
        elo_math.apply_scale(1500, "fide")
