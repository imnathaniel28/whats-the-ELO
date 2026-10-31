"""Compare this service's prediction against the original torch pipeline.

The unit tests already check the board encoder against a transcription of the
original algorithm. This script goes further and runs the real torch model from
the original repo on the same game, which is the only way to confirm the ONNX
graph and the stepping loop agree with it end to end.

Needs the original repo plus `pip install '.[export]'`:

    python scripts/parity_check.py \
        --repo /path/to/Ai-Guess-the-elo \
        --pgn  tests/data/game.pgn

Pass --nodes to make both paths deterministic. With a wall-clock limit the two
runs search different numbers of nodes and the Elo readings will differ by tens
of points even though both are correct.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="checkout of the original repo")
    parser.add_argument("--pgn", default="tests/data/game.pgn")
    parser.add_argument("--nodes", type=int, default=200_000, help="nodes per position")
    parser.add_argument("--max-plies", type=int, default=None)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import chess.engine
    import numpy as np
    import torch

    from elo_ai.helper_functions import game_analysis, position_converters
    from elo_ai.helper_functions.elo_range import get_elo_prediction, get_rating_ranges
    from elo_ai.models import complex_network

    from whats_the_elo import features, predictor
    from whats_the_elo.config import get_settings
    from whats_the_elo.elo_math import elo_from_probabilities, round_elo

    settings = get_settings()
    if not settings.engine_path:
        raise SystemExit("no Stockfish found; set STOCKFISH_PATH")
    pgn_text = Path(args.pgn).read_text()
    limit = chess.engine.Limit(nodes=args.nodes)

    # --- this service -------------------------------------------------------
    ours = predictor.predict(
        pgn_text, node_limit=args.nodes, max_plies=args.max_plies, include_probabilities=True
    )
    print(f"service : white={ours['white_elo']:5d}  black={ours['black_elo']:5d}")

    # --- the original torch pipeline, on features built the same way --------
    game = features.parse_pgn(pgn_text)
    engine = chess.engine.SimpleEngine.popen_uci(settings.engine_path)
    try:
        # Patch the original's hardcoded Limit(time=...) to the node budget so
        # both paths see identical engine output.
        original_analyse = engine.analyse

        def analyse(board, _limit, **kwargs):
            return original_analyse(board, limit, **kwargs)

        engine.analyse = analyse
        torch_analysis = game_analysis.analyze_game(game, engine)
    finally:
        engine.quit()

    torch_positions, _ = position_converters.convert_position(
        game, position_converters.fen_to_board_mirror
    )

    model = complex_network.EloGuesser(
        17, input_channels=2, num_classes=len(get_rating_ranges())
    )
    weights = repo / "elo_ai/models/rating_ranges/boards_mirrors.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    hidden = cell = None
    prediction = None
    with torch.no_grad():
        for ply in range(torch_analysis.size(1)):
            prediction, (hidden, cell) = model(
                (
                    torch_positions[:, ply].unsqueeze(1),
                    torch_analysis[:, ply].unsqueeze(1),
                ),
                hidden,
                cell,
            )
    torch_elo = get_elo_prediction(prediction, False, round=True)
    print(f"original: white={torch_elo[0]:5d}  black={torch_elo[1]:5d}")

    # --- compare ------------------------------------------------------------
    torch_probs = prediction.detach().numpy()
    ours_probs = np.array(
        [ours["white_probabilities"], ours["black_probabilities"]], dtype=np.float32
    )
    probability_error = float(np.max(np.abs(torch_probs - ours_probs)))
    print(f"\nmax abs probability difference: {probability_error:.3e}")
    for side, index in (("white", 0), ("black", 1)):
        theirs = round_elo(elo_from_probabilities(torch_probs[index]))
        mine = ours[f"{side}_elo"]
        print(f"  {side}: original={theirs} service={mine} delta={abs(theirs - mine)}")

    assert probability_error < 1e-3, "ONNX path diverges from the torch original"
    print("\nOK: the service agrees with the original pipeline")


if __name__ == "__main__":
    main()
