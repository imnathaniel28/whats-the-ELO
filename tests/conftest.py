import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def pgn_text():
    return (DATA_DIR / "game.pgn").read_text()


@pytest.fixture(scope="session")
def short_pgn_text():
    return (DATA_DIR / "game2v2.pgn").read_text()


@pytest.fixture(scope="session")
def has_engine():
    from whats_the_elo.config import get_settings

    return get_settings().engine_path is not None


@pytest.fixture(autouse=True)
def _clean_settings():
    """Keep env-var changes in one test from leaking into the next."""
    from whats_the_elo.config import reset_settings

    reset_settings()
    yield
    reset_settings()


def pytest_collection_modifyitems(config, items):
    """Skip engine-marked tests when no Stockfish is available."""
    import shutil

    available = bool(os.environ.get("STOCKFISH_PATH")) or bool(
        shutil.which("stockfish") or shutil.which("stockfish.exe")
    )
    if available:
        return
    skip = pytest.mark.skip(reason="no Stockfish; set STOCKFISH_PATH to run")
    for item in items:
        if "engine" in item.keywords:
            item.add_marker(skip)
