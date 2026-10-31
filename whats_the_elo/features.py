"""Feature extraction: PGN + Stockfish -> the two tensors the model consumes.

Numpy port of the original `elo_ai/helper_functions/position_converters.py` and
`game_analysis.py`, with the torch dependency removed. `scripts/parity_check.py`
asserts this produces bit-identical positions to the torch original.

Two arrays come out of `extract_features`, both with a leading axis of 2 for
(white, black):

    positions  float32 [2, seq, 2, 8, 8]   board values, plus a turn-mask channel
    analysis   float32 [2, seq, 17]        Stockfish evaluation features

`seq` is the number of plies *per side*: the full ply list is padded to an even
length and then dealt alternately, so white gets plies 0, 2, 4 ... and black gets
1, 3, 5 ...
"""
import io

import chess
import chess.engine
import chess.pgn
import numpy as np

MATE_SCORE = 1_000
# Width of the multipv request; the first 10 slots of the feature vector.
NOF_MOVES = 10
NORMALIZER = 20.0
FEATURE_SIZE = 17

# Piece values seen from the side to move. When black is to move the board is
# rotated 180 degrees and the signs flip, so the network always sees the position
# from the perspective of the player whose move is being judged.
_WHITE_MAP = {
    "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
    "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
}
_BLACK_MAP = {piece: -value for piece, value in _WHITE_MAP.items()}


class FeatureError(ValueError):
    """The PGN could not be turned into model input."""


# ---------------------------------------------------------------------------
# Position encoding
# ---------------------------------------------------------------------------
def fen_to_board_mirror(fen):
    """Encode a FEN as an 8x8 array of normalized piece values.

    Mirrored for black-to-move: reversing the placement string rotates the board
    180 degrees, which is why the original uses a plain string reverse here.
    """
    board = np.zeros((8, 8), dtype=np.float32)
    parts = fen.split(" ")
    if len(parts) < 2:
        raise FeatureError(f"malformed FEN: {fen!r}")
    placement, side_to_move = parts[0], parts[1]

    if side_to_move == "w":
        mappings = _WHITE_MAP
    else:
        mappings = _BLACK_MAP
        placement = placement[::-1]

    row, col = 0, 0
    for char in placement:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row, col] = mappings[char] / NORMALIZER
            col += 1
    return board


def _split_streams(per_ply, pad_value):
    """Pad to an even ply count, then deal alternately into (white, black)."""
    if len(per_ply) % 2:
        per_ply = per_ply + [np.full_like(per_ply[0], pad_value)]
    return np.stack((np.stack(per_ply[0::2]), np.stack(per_ply[1::2])))


def encode_positions(game, max_plies=None):
    """Return float32 [2, seq, 2, 8, 8] for the game's mainline."""
    board = game.board()
    boards = []
    for ply, move in enumerate(game.mainline_moves()):
        if max_plies is not None and ply >= max_plies:
            break
        boards.append(fen_to_board_mirror(board.fen()))
        board.push(move)

    if not boards:
        raise FeatureError("PGN contains no moves")

    # Positions pad with zeros (the original uses torch.zeros_like here).
    stacked = _split_streams(boards, 0.0)  # [2, seq, 8, 8]

    # Channel 1 is a constant turn mask: +1 on white's stream, -1 on black's.
    white_mask = np.ones_like(stacked[0])
    black_mask = -np.ones_like(stacked[1])
    white = np.stack((stacked[0], white_mask), axis=1)
    black = np.stack((stacked[1], black_mask), axis=1)
    return np.stack((white, black)).astype(np.float32)  # [2, seq, 2, 8, 8]


# ---------------------------------------------------------------------------
# Engine analysis
# ---------------------------------------------------------------------------
def _analysis_before_move(engine, board, limit):
    """Top-N scores and the win/draw/loss split for the position before a move."""
    infos = engine.analyse(board, limit, multipv=NOF_MOVES)
    if not infos:
        raise FeatureError(f"engine returned no analysis for {board.fen()!r}")

    scores = [info["score"].relative.score(mate_score=MATE_SCORE) for info in infos]
    # Fewer legal moves than NOF_MOVES (or a terminal position, where Stockfish
    # answers with a single line) pads with -MATE_SCORE; truncate in case the
    # engine ever returns more lines than requested.
    scores = scores[:NOF_MOVES]
    scores += [-MATE_SCORE] * (NOF_MOVES - len(scores))
    top_moves = [score / MATE_SCORE for score in scores]

    wdl = infos[0]["score"].relative.wdl()
    before_wdl = [wdl.winning_chance(), wdl.drawing_chance(), wdl.losing_chance()]

    principal_variation = infos[0].get("pv")
    best_move = principal_variation[0].uci() if principal_variation else ""
    return top_moves, before_wdl, best_move


def _analysis_after_move(engine, board, limit):
    """Score and win/draw/loss split after the move, from the mover's side."""
    info = engine.analyse(board, limit)
    # The opponent is to move now, so negate to stay on the mover's perspective.
    score = info["score"].relative.score(mate_score=MATE_SCORE) * -1 / MATE_SCORE
    wdl = info["score"].relative.wdl()
    # Reversed relative to `before`, for the same perspective-flip reason.
    after_wdl = [wdl.losing_chance(), wdl.drawing_chance(), wdl.winning_chance()]
    return [score], after_wdl


def analyze_game(game, engine, limit, max_plies=None, progress=None):
    """Return (analysis [2, seq, 17], best_moves) for the game's mainline."""
    board = game.board()
    rows = []
    best_moves = []
    moves = list(game.mainline_moves())
    if max_plies is not None:
        moves = moves[:max_plies]
    if not moves:
        raise FeatureError("PGN contains no moves")

    for index, move in enumerate(moves):
        top_moves, before_wdl, best_move = _analysis_before_move(engine, board, limit)
        best_moves.append(best_move)
        board.push(move)
        after_score, after_wdl = _analysis_after_move(engine, board, limit)

        row = np.asarray(top_moves + before_wdl + after_score + after_wdl, dtype=np.float32)
        if row.shape[0] != FEATURE_SIZE:
            raise FeatureError(f"expected {FEATURE_SIZE} features, built {row.shape[0]}")
        rows.append(row)

        if progress is not None:
            progress(index + 1, len(moves))

    # Unlike positions, the analysis pad value is -MATE_SCORE, not zero.
    analysis = _split_streams(rows, float(-MATE_SCORE))
    return analysis.astype(np.float32), best_moves


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_pgn(pgn_text):
    """Parse the first game out of a PGN string."""
    if not pgn_text or not pgn_text.strip():
        raise FeatureError("PGN is empty")
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception as exc:  # python-chess is lenient, but tolerate surprises
        raise FeatureError(f"could not parse PGN: {exc}") from exc
    if game is None:
        raise FeatureError("could not parse a game from the PGN")
    if game.errors:
        # Illegal/ambiguous moves are reported here rather than raised. The game
        # is still usable up to the bad move, but silently scoring a truncated
        # game would be misleading.
        raise FeatureError(f"PGN contains invalid moves: {game.errors[0]}")
    if not list(game.mainline_moves()):
        raise FeatureError("PGN contains no moves")
    return game


def extract_features(game, engine, limit, max_plies=None, progress=None):
    """Build both model inputs plus the per-move metadata the caller may want."""
    analysis, best_moves = analyze_game(
        game, engine, limit, max_plies=max_plies, progress=progress
    )
    positions = encode_positions(game, max_plies=max_plies)

    if positions.shape[1] != analysis.shape[1]:
        raise FeatureError(
            f"position/analysis length mismatch: {positions.shape[1]} vs {analysis.shape[1]}"
        )

    return {
        "positions": positions,
        "analysis": analysis,
        "best_moves": best_moves,
        "plies_analyzed": len(best_moves),
        "headers": dict(game.headers),
    }


def game_fens(game, max_plies=None):
    """FEN after every ply, starting from the initial position."""
    board = game.board()
    fens = [board.fen()]
    for ply, move in enumerate(game.mainline_moves()):
        if max_plies is not None and ply >= max_plies:
            break
        board.push(move)
        fens.append(board.fen())
    return fens
