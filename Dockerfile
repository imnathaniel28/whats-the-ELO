# Stockfish comes from the distro rather than the repo: the upstream binary is
# ~114 MB (over GitHub's file limit) and is GPLv3, so the image installs it and
# the repo stays MIT.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y stockfish \
    && rm -rf /var/lib/apt/lists/*

# Debian installs the binary here; config.discover_engine would also find it on
# PATH, but being explicit means a broken PATH shows up at build time.
ENV STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app

COPY pyproject.toml README.md ./
COPY whats_the_elo ./whats_the_elo
RUN pip install --no-cache-dir ".[service]"

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()"

# One worker per core is wrong here: each request holds a Stockfish process for
# seconds. Scale with ELO_MAX_CONCURRENT and replicas instead.
CMD ["uvicorn", "whats_the_elo.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
