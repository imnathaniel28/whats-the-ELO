"""Re-export whats_the_elo/model.onnx from the original PyTorch weights.

Only needed if the model is retrained. The committed model.onnx was produced by
this script, so running it is not part of normal use — it is here so the
artifact is reproducible rather than mysterious.

Needs the original training repo (for `elo_ai`) plus `pip install '.[export]'`:

    python scripts/export_onnx.py \
        --weights /path/to/Ai-Guess-the-elo/elo_ai/models/rating_ranges/boards_mirrors.pt \
        --repo    /path/to/Ai-Guess-the-elo \
        --out     whats_the_elo/model.onnx

The graph is one LSTM timestep wide with (h, c) threaded through as explicit
inputs, because inference walks the game ply by ply:

    inputs:  position [2,1,2,8,8]  evaluation [2,1,17]  h0 [1,2,64]  c0 [1,2,64]
    outputs: probs    [2,39]       hn         [1,2,64]  cn [1,2,64]

The leading 2 is (white, black). It is fixed, not a batch axis you can grow:
the original `EloGuesser.forward` special-cases `position.size(0) == 2` and runs
the CNN separately per side.
"""
import argparse
import sys
from pathlib import Path

INPUT_SIZE = 17
CHANNELS = 2


def _build_wrapper(torch, model):
    """Flatten EloGuesser's (out, (hn, cn)) into three named ONNX outputs.

    Nested so that `torch` only has to be importable when this script runs.
    """

    class _SingleStep(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, position, evaluation, h0, c0):
            out, (hn, cn) = self.inner((position, evaluation), h0, c0)
            return out, hn, cn

    return _SingleStep(model).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="boards_mirrors.pt")
    parser.add_argument(
        "--repo", required=True, help="checkout of the original repo, for the elo_ai package"
    )
    parser.add_argument("--out", default="whats_the_elo/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))

    import numpy as np
    import torch
    from elo_ai.models.complex_network import LSTM_HIDDEN_SIZE, NUM_LAYERS, EloGuesser

    from whats_the_elo.elo_math import NUM_CLASSES

    model = EloGuesser(INPUT_SIZE, input_channels=CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    wrapper = _build_wrapper(torch, model)

    position = torch.randn(2, 1, CHANNELS, 8, 8)
    evaluation = torch.randn(2, 1, INPUT_SIZE)
    hidden = torch.zeros(NUM_LAYERS, 2, LSTM_HIDDEN_SIZE)
    cell = torch.zeros(NUM_LAYERS, 2, LSTM_HIDDEN_SIZE)

    with torch.no_grad():
        reference = wrapper(position, evaluation, hidden, cell)

    torch.onnx.export(
        wrapper,
        (position, evaluation, hidden, cell),
        args.out,
        input_names=["position", "evaluation", "h0", "c0"],
        output_names=["probs", "hn", "cn"],
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported -> {args.out}")

    import onnxruntime as ort

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = session.run(
        ["probs", "hn", "cn"],
        {
            "position": position.numpy(),
            "evaluation": evaluation.numpy(),
            "h0": hidden.numpy(),
            "c0": cell.numpy(),
        },
    )
    for name, expected, actual in zip(("probs", "hn", "cn"), reference, got):
        error = float(np.max(np.abs(expected.numpy() - actual)))
        print(f"  max abs err {name}: {error:.3e}")
        assert error < 1e-4, f"ONNX/torch parity failed for {name}"
    print("OK: ONNX matches torch")


if __name__ == "__main__":
    main()
