# What's the ELO

Predicts both players' Elo from a single chess game. PGN in, two ratings out.

This is the headless core of [Ai-Guess-the-elo](https://github.com/HliasOuzounis/Ai-Guess-the-elo),
repackaged to be embedded in a chess site or app. There is no UI, no templates,
no static files, no desktop window — every response is JSON.

```bash
curl -s localhost:8000/v1/predict -H 'Content-Type: application/json' \
  -d '{"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ..."}'
```

```json
{
  "white_elo": 1700,
  "black_elo": 1950,
  "white_elo_raw": 1719.2,
  "black_elo_raw": 1938.6,
  "scale": "lichess",
  "num_moves": 67,
  "plies_analyzed": 67,
  "truncated": false,
  "headers": { "White": "...", "WhiteElo": "1714" },
  "analysis_seconds": 8.21
}
```

## How it works, and what that costs you

The model is a CNN over the board plus a dense branch over Stockfish's
evaluation, joined into an LSTM that is stepped one ply at a time. It outputs a
probability across 39 rating bands (0–3900 in 100-point steps); the reported Elo
is the probability-weighted mean of the band midpoints.

**Stockfish is required at inference time.** All 17 evaluation features per ply
come from the engine — there is no engine-free path. The service makes two
`analyse` calls per ply, so:

```
wall clock  ≈  2 × plies × ELO_TIME_LIMIT
```

A 67-ply game at the default 0.1s is roughly **8–13 seconds**. Model inference
itself is ~3 ms; essentially all the latency is Stockfish. Plan the integration
around that: queue it as a job, or show a progress indicator. Do not call it
synchronously from a page render.

## Accuracy

From the original project's evaluation on 3000 held-out games:

| Within | Share of predictions |
| --- | --- |
| ±200 Elo | 60.5% |
| ±400 Elo | 87.7% |
| ±600 Elo | 96.6% |

It overestimates games below 1000 and underestimates above 2600. One game is a
weak signal for a player's rating, and each player's prediction is influenced by
their opponent through the positions that arise. Present it as a guess, not a
measurement.

## Running it

### Docker (recommended)

The image installs Stockfish itself.

```bash
docker compose up --build
curl localhost:8000/healthz
```

### Directly

Needs a Stockfish binary. `apt install stockfish`, `brew install stockfish`, or
download from [stockfishchess.org](https://stockfishchess.org/download/).

```bash
pip install '.[service]'
export STOCKFISH_PATH=/usr/games/stockfish   # or leave unset if it's on PATH
uvicorn whats_the_elo.api:app --port 8000
```

### As a Python library

If your backend is already Python, skip HTTP entirely:

```python
from whats_the_elo import predict

result = predict(pgn_text, scale="lichess")
print(result["white_elo"], result["black_elo"])
```

`predict()` is synchronous and blocks for seconds. Call it from a worker, not a
request handler.

## API

### `POST /v1/predict`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `pgn` | string | required | Only the first game in the PGN is scored. |
| `scale` | `"lichess"` \| `"chesscom"` | `"lichess"` | The model is trained on lichess ratings; `chesscom` applies the published blitz conversion, roughly 400 points lower. |
| `time_limit` | float | `0.1` | Seconds of Stockfish per position, 0.01–2.0. |
| `node_limit` | int | — | Nodes per position instead of a time limit. **Use this if you need reproducible output** (see below). |
| `max_plies` | int | — | Stop after this many plies. Caps latency on long games. |
| `include_probabilities` | bool | `false` | Adds the 39-bin distribution per player, plus `rating_bins`. |
| `include_trajectory` | bool | `false` | Adds the running Elo estimate after every ply — what the original's visualizer plotted. |
| `include_fens` | bool | `false` | Adds the FEN after every ply. |
| `include_best_moves` | bool | `false` | Adds Stockfish's best move in each position, as UCI. |

Errors: `422` unparseable PGN or bad parameters, `413` PGN over the size cap,
`503` no Stockfish or all analysis slots busy, `500` otherwise.

### `POST /v1/validate`

`{"pgn": "..."}` → move count, headers, and `estimated_seconds`. No engine work,
so it returns instantly. Use it to validate input and to warn the user how long
the real call will take.

### `GET /v1/info`

Model metadata, rating bins, accuracy figures, and this instance's effective limits.

### `GET /healthz` · `GET /healthz/engine`

`/healthz` is cheap and checks the model loaded and a Stockfish binary was
found; it returns `503` when either is missing. `/healthz/engine` actually
starts Stockfish and reports its version — slower, so don't use it as a
liveness probe.

## Configuration

All environment variables. Nothing here reads a config file.

| Variable | Default | Notes |
| --- | --- | --- |
| `STOCKFISH_PATH` | auto-discovered on `PATH` | Absolute path to the binary. |
| `ELO_TIME_LIMIT` | `0.1` | Seconds per position. The value the model was trained against. |
| `ELO_NODE_LIMIT` | unset | Nodes per position instead of time. Makes output reproducible. |
| `ELO_MAX_CONCURRENT` | half the CPU count | Concurrent analyses, each one Stockfish process. |
| `ELO_ENGINE_THREADS` | `1` | UCI `Threads` per engine. Keep at 1 and scale with concurrency. |
| `ELO_ENGINE_HASH_MB` | `16` | UCI `Hash` per engine. |
| `ELO_MAX_PLIES` | `300` | Hard cap, overrides a larger `max_plies` in a request. |
| `ELO_MAX_PGN_BYTES` | `262144` | Request body cap for the PGN field. |
| `ELO_ANALYSIS_TIMEOUT` | `300` | Seconds to wait for a free analysis slot. |
| `ELO_API_KEY` | unset | When set, requests must send a matching `X-API-Key`. |
| `ELO_CORS_ORIGINS` | unset | Comma-separated origins. Only needed if browsers call the service directly. |
| `ELO_DOCS` | `false` | Set to `1` to expose `/docs` and `/openapi.json`. |
| `ELO_MODEL_PATH` | packaged `model.onnx` | Override to load different weights. |
| `ELO_ONNX_THREADS` | `1` | onnxruntime thread count. Inference is ~3 ms; 1 is plenty. |

### Determinism

With a time limit, Stockfish searches a different number of nodes on every run,
so **the same game can return a different Elo each time** — tens of points of
drift. If you cache results, show them back to users, or test against expected
values, set `ELO_NODE_LIMIT` (or pass `node_limit`) instead. `200000` nodes is
roughly comparable to 0.1s on a modern core.

### Scaling

Each in-flight request holds a Stockfish process for seconds, so this is
CPU-bound, not IO-bound. Run one uvicorn worker, set `ELO_MAX_CONCURRENT` to the
cores you want to spend, and add replicas for more throughput. Requests beyond
the limit wait for a slot and get `503` after `ELO_ANALYSIS_TIMEOUT`.

### Security

The service has no authentication unless you set `ELO_API_KEY`. Keep it on an
internal network and call it from your own backend, or set the key. A long
analysis is also a cheap way for someone to burn your CPU — keep
`ELO_MAX_PLIES` and `ELO_MAX_CONCURRENT` tight and rate-limit at your edge.

## Development

```bash
pip install -e '.[service,dev]'
export STOCKFISH_PATH=/usr/games/stockfish
pytest
```

Tests that need an engine are marked `engine` and skip automatically when no
Stockfish is found, so `pytest` passes on a bare checkout.

`scripts/parity_check.py` runs this service and the original torch pipeline on
the same game and asserts their output distributions match; it needs a checkout
of the upstream repo. `scripts/export_onnx.py` regenerates
`whats_the_elo/model.onnx` from the original `.pt` weights.

## License and attribution

MIT. The model and the training work are Ilias Ouzounis's — see
[NOTICE.md](NOTICE.md) for what was carried over from upstream and what changed.

Two licenses to be aware of before you ship a binary: **Stockfish is GPLv3**
(which is why it is never vendored here), and **python-chess is GPLv3** (and is
imported directly). Hosting this as a service you run yourself is not
distribution; bundling it into a shipped app is. Details in
[NOTICE.md](NOTICE.md).
