"""Feature-extraction tests.

The board encoder is the part most likely to drift from the trained weights, so
it is checked two ways: against a verbatim transcription of the original
project's algorithm, and against hardcoded values for known positions.
"""
import chess
import numpy as np
import pytest

from whats_the_elo import features


# Verbatim transcription of elo_ai/helper_functions/position_converters.py's
# fen_to_board_mirror, with the torch tensor swapped for a list. Kept
# deliberately un-refactored: its job is to be the original, not to be tidy.
def _reference_fen_to_board_mirror(fen_str):
    board = [[0.0] * 8 for _ in range(8)]
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    normalizer = 20
    mappings = {
        "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
        "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
    } if move == "w" else {
        "P": -1, "N": -3, "B": -3.5, "R": -5, "Q": -9, "K": -20,
        "p": 1, "n": 3, "b": 3.5, "r": 5, "q": 9, "k": 20,
    }
    fen = fen if move == "w" else fen[::-1]
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row][col] = mappings[char] / normalizer
            col += 1
    return board


def test_encoder_matches_the_original_over_a_real_game(pgn_text):
    game = features.parse_pgn(pgn_text)
    board = game.board()
    checked = 0
    for move in game.mainline_moves():
        fen = board.fen()
        expected = np.array(_reference_fen_to_board_mirror(fen), dtype=np.float32)
        actual = features.fen_to_board_mirror(fen)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        checked += 1
        board.push(move)
    assert checked > 20, "fixture should cover a decent number of positions"


def test_encoder_matches_the_original_across_varied_positions():
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "r1bq1r1k/ppp2ppp/2n5/3p4/1b1P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 10",
        "8/P7/8/8/8/8/7p/K6k b - - 0 1",
    ]
    for fen in fens:
        expected = np.array(_reference_fen_to_board_mirror(fen), dtype=np.float32)
        np.testing.assert_allclose(features.fen_to_board_mirror(fen), expected, atol=0)


def test_start_position_encoding_is_as_expected():
    board = features.fen_to_board_mirror(chess.STARTING_FEN)
    assert board.shape == (8, 8)
    # White is to move, so white's pieces are the positive ones.
    assert board[7, 4] == pytest.approx(1.0)    # white king, 20/20
    assert board[0, 4] == pytest.approx(-1.0)   # black king
    assert board[6, 0] == pytest.approx(0.05)   # white pawn, 1/20
    assert board[1, 0] == pytest.approx(-0.05)  # black pawn
    np.testing.assert_allclose(board[2:6], 0.0)


def test_side_to_move_always_sees_its_own_pieces_as_positive():
    # This is the whole point of the mirroring: the network judges the player
    # whose turn it is, always from that player's perspective.
    white_to_move = features.fen_to_board_mirror(chess.STARTING_FEN)
    black_to_move = features.fen_to_board_mirror(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    # Rank 8 in the array's row 0 belongs to the opponent in both cases.
    assert white_to_move[0].sum() < 0
    assert black_to_move[0].sum() < 0
    assert white_to_move[7].sum() > 0
    assert black_to_move[7].sum() > 0


def test_malformed_fen_is_rejected():
    with pytest.raises(features.FeatureError, match="malformed FEN"):
        features.fen_to_board_mirror("rnbqkbnr")


# ---------------------------------------------------------------------------
# Stream splitting and padding
# ---------------------------------------------------------------------------
def test_positions_shape_and_turn_mask(pgn_text):
    game = features.parse_pgn(pgn_text)
    positions = features.encode_positions(game)

    total_plies = sum(1 for _ in game.mainline_moves())
    expected_seq = (total_plies + 1) // 2

    assert positions.shape == (2, expected_seq, 2, 8, 8)
    assert positions.dtype == np.float32
    # Channel 1 is the constant turn mask: +1 on white's stream, -1 on black's.
    np.testing.assert_allclose(positions[0, :, 1], 1.0)
    np.testing.assert_allclose(positions[1, :, 1], -1.0)


def test_odd_ply_count_pads_white_and_black_to_equal_length():
    # 3 plies: white gets plies 0 and 2, black gets ply 1 plus a zero pad.
    game = features.parse_pgn('[Event "t"]\n\n1. e4 e5 2. Nf3 *\n')
    positions = features.encode_positions(game)
    assert positions.shape == (2, 2, 2, 8, 8)
    # The pad is zeros in the board channel (the analysis pad differs; see below).
    np.testing.assert_allclose(positions[1, 1, 0], 0.0)


def test_max_plies_truncates():
    game = features.parse_pgn('[Event "t"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *\n')
    assert features.encode_positions(game).shape[1] == 3
    assert features.encode_positions(game, max_plies=4).shape[1] == 2


# ---------------------------------------------------------------------------
# PGN parsing
# ---------------------------------------------------------------------------
def test_valid_pgn_parses(pgn_text):
    game = features.parse_pgn(pgn_text)
    assert game.headers["WhiteElo"] == "1714"
    assert sum(1 for _ in game.mainline_moves()) == 67


@pytest.mark.parametrize(
    "bad,message",
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ('[Event "t"]\n\n*\n', "no moves"),
        ('[Event "t"]\n\n1. e4 e5 2. Qxf7 *\n', "invalid moves"),
    ],
)
def test_bad_pgn_is_rejected(bad, message):
    with pytest.raises(features.FeatureError, match=message):
        features.parse_pgn(bad)


def test_fens_cover_every_ply(pgn_text):
    game = features.parse_pgn(pgn_text)
    fens = features.game_fens(game)
    assert len(fens) == 68  # 67 plies plus the starting position
    assert fens[0] == chess.STARTING_FEN
