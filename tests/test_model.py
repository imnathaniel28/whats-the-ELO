"""Model-graph tests. These need no Stockfish: the features are synthesised."""
import numpy as np
import pytest

from whats_the_elo import predictor
from whats_the_elo.elo_math import NUM_CLASSES


def _fake_features(seq):
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((2, seq, 2, 8, 8)).astype(np.float32)
    analysis = rng.standard_normal((2, seq, 17)).astype(np.float32)
    return positions, analysis


def test_graph_signature_matches_what_the_stepping_loop_feeds():
    session = predictor.get_session()
    inputs = {i.name: i.shape for i in session.get_inputs()}
    outputs = {o.name: o.shape for o in session.get_outputs()}

    assert inputs["position"] == [2, 1, 2, 8, 8]
    assert inputs["evaluation"] == [2, 1, 17]
    assert inputs["h0"] == [1, 2, 64]
    assert inputs["c0"] == [1, 2, 64]
    assert outputs["probs"] == [2, NUM_CLASSES]
    assert outputs["hn"] == [1, 2, 64]
    assert outputs["cn"] == [1, 2, 64]


def test_output_is_a_probability_distribution_per_player():
    positions, analysis = _fake_features(12)
    probabilities, trajectory = predictor.run_model(positions, analysis)

    assert probabilities.shape == (2, NUM_CLASSES)
    assert trajectory is None
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0], atol=1e-5)
    assert (probabilities >= 0).all()


def test_trajectory_has_one_entry_per_ply():
    positions, analysis = _fake_features(7)
    probabilities, trajectory = predictor.run_model(
        positions, analysis, keep_trajectory=True
    )
    assert len(trajectory) == 7
    # The last trajectory entry is the returned prediction.
    np.testing.assert_allclose(trajectory[-1], probabilities)


def test_lstm_state_actually_carries_across_plies():
    # If (h, c) were dropped between steps, a longer game would give the same
    # answer as a one-ply game built from the same final position.
    positions, analysis = _fake_features(10)
    full, _ = predictor.run_model(positions, analysis)
    last_only, _ = predictor.run_model(positions[:, -1:], analysis[:, -1:])
    assert not np.allclose(full, last_only, atol=1e-4)


def test_single_ply_game_is_scoreable():
    positions, analysis = _fake_features(1)
    probabilities, _ = predictor.run_model(positions, analysis)
    assert probabilities.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize(
    "positions_shape,analysis_shape",
    [
        ((1, 5, 2, 8, 8), (2, 5, 17)),  # wrong player axis
        ((2, 5, 2, 8), (2, 5, 17)),     # missing a board dimension
    ],
)
def test_wrong_input_shapes_are_rejected(positions_shape, analysis_shape):
    positions = np.zeros(positions_shape, dtype=np.float32)
    analysis = np.zeros(analysis_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        predictor.run_model(positions, analysis)


def test_session_is_shared_across_calls():
    predictor.reset_session()
    first = predictor.get_session()
    assert predictor.get_session() is first
