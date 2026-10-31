"""Stockfish process management.

Stockfish is mandatory: all 17 evaluation features come from it, so there is no
engine-free path. It is spoken to over UCI on stdio, so any build works — the
service never ships a binary, it locates one (see `config.discover_engine`).

A fresh process is spawned per analysis rather than pooled. Spawning costs tens
of milliseconds against multiple seconds of analysis, and it means a crashed or
wedged engine can never be handed to the next request. Concurrency is capped by
a semaphore so N in-flight requests cannot fork N Stockfish processes and starve
the box.
"""
import threading
from contextlib import contextmanager

import chess.engine

from .config import get_settings


class EngineUnavailable(RuntimeError):
    """No usable Stockfish binary, or it failed to start."""


class EngineBusy(RuntimeError):
    """All analysis slots are occupied."""


_semaphores = {}
_semaphore_lock = threading.Lock()


def _semaphore(limit):
    """One semaphore per configured limit, created lazily."""
    with _semaphore_lock:
        if limit not in _semaphores:
            _semaphores[limit] = threading.BoundedSemaphore(limit)
        return _semaphores[limit]


def search_limit(time_limit=None, node_limit=None):
    """Build the per-position search budget.

    A node limit is reproducible; a time limit is not, because Stockfish will
    search a different number of nodes on every run and on every machine. The
    model was trained against a 0.1s time limit, so that remains the default.
    """
    settings = get_settings()
    nodes = node_limit if node_limit is not None else settings.node_limit
    if nodes:
        return chess.engine.Limit(nodes=int(nodes))
    seconds = time_limit if time_limit is not None else settings.time_limit
    return chess.engine.Limit(time=float(seconds))


@contextmanager
def analysis_engine(acquire_timeout=None):
    """Yield a configured Stockfish, holding a concurrency slot for the duration."""
    settings = get_settings()
    if not settings.engine_path:
        raise EngineUnavailable(
            "Stockfish not found. Set STOCKFISH_PATH to the binary, or install "
            "it so that 'stockfish' is on PATH."
        )

    timeout = settings.analysis_timeout if acquire_timeout is None else acquire_timeout
    slot = _semaphore(settings.max_concurrent)
    if not slot.acquire(timeout=timeout):
        raise EngineBusy(
            f"all {settings.max_concurrent} analysis slots busy; retry shortly"
        )

    engine = None
    try:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(settings.engine_path)
        except Exception as exc:
            raise EngineUnavailable(
                f"could not start Stockfish at {settings.engine_path!r}: {exc}"
            ) from exc

        # Keep each process single-threaded by default: concurrency comes from
        # running several requests, not from one engine eating every core.
        for name, value in (
            ("Threads", settings.engine_threads),
            ("Hash", settings.engine_hash_mb),
        ):
            try:
                engine.configure({name: value})
            except Exception:
                # Older or patched builds may not expose the option; the default
                # is fine and not worth failing the request over.
                pass

        yield engine
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                # The process may already be gone; make sure it cannot linger.
                try:
                    engine.close()
                except Exception:
                    pass
        slot.release()


def engine_version():
    """Identify the located engine, or return None if it cannot be started."""
    settings = get_settings()
    if not settings.engine_path:
        return None
    try:
        with chess.engine.SimpleEngine.popen_uci(settings.engine_path) as engine:
            return engine.id.get("name")
    except Exception:
        return None
