# Attribution and third-party licenses

## Upstream project

This repository is a headless extraction of **Ai-Guess-the-elo** by Ilias Ouzounis
(<https://github.com/HliasOuzounis/Ai-Guess-the-elo>), MIT licensed. The neural
network architecture, the training work, and the trained weights in
`whats_the_elo/model.onnx` are that project's; this repository repackages the
inference path as a service and adds no new model work.

What was carried over, and what changed:

| From upstream | Here | Change |
| --- | --- | --- |
| `elo_ai/models/rating_ranges/boards_mirrors.pt` | `whats_the_elo/model.onnx` | Exported to ONNX so the service needs no PyTorch |
| `elo_ai/helper_functions/position_converters.py` | `whats_the_elo/features.py` | Numpy instead of torch; inference path only |
| `elo_ai/helper_functions/game_analysis.py` | `whats_the_elo/features.py` | Numpy instead of torch; `tqdm` replaced by a callback |
| `elo_ai/helper_functions/elo_range.py` | `whats_the_elo/elo_math.py` | Numpy instead of torch; inference path only |
| `guess_the_elo.py` | `whats_the_elo/predictor.py` | Model loaded once per process instead of once per game; no argparse |
| `chess_analyzer_app.py` | `whats_the_elo/api.py` | FastAPI instead of Flask; returns ratings as JSON instead of opening a desktop window |

Deliberately not carried over: the tkinter visualizer, the matplotlib plotting,
the training scripts and notebook, the dataset tooling, the PyInstaller desktop
demo, the Android app, and all images and UI assets.

## Stockfish

The service requires Stockfish at runtime but **does not include it**.
Stockfish is licensed **GPLv3**, which is why it is installed by the container
(`apt-get install stockfish`) or located on the host `PATH` rather than vendored
here. That keeps this repository MIT and leaves the GPLv3 obligations with the
distribution that ships the binary.

If you later choose to bundle a Stockfish binary with your own application, the
GPLv3 terms apply to that distribution — including for the Android `.so` and any
desktop installer. Check them before shipping.

Stockfish: <https://github.com/official-stockfish/Stockfish>

## Runtime dependencies

| Package | License |
| --- | --- |
| numpy | BSD-3-Clause |
| onnxruntime | MIT |
| python-chess (`chess`) | GPLv3 |
| fastapi | MIT |
| uvicorn | BSD-3-Clause |

Note that **python-chess is GPLv3**. It is imported directly by this service, so
if you distribute a binary that embeds this code and python-chess, the GPLv3
terms apply to that distribution. Running it as a network service you host
yourself does not trigger distribution.
