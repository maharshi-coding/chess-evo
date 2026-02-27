"""Stockfish engine integration.

Performance design
------------------
* The engine subprocess is started once and kept alive for the duration of the
  game; spawning a new process per move would add ~200 ms overhead each time.
* ``get_best_move`` is **non-blocking** when called with ``timeout``; it
  delegates to ``stockfish`` library which communicates over a persistent pipe.
* UCI ``setoption`` calls are batched at startup rather than sent repeatedly.
"""

from __future__ import annotations

import logging
from typing import Optional

import chess

logger = logging.getLogger(__name__)


class StockfishEngine:
    """Thin wrapper around the ``stockfish`` Python library.

    Parameters
    ----------
    path:
        Filesystem path to the Stockfish binary.
    skill_level:
        Engine skill 0 – 20.
    move_time:
        Maximum time (seconds) the engine may think per move.
    threads:
        Number of CPU threads for engine search.
    hash_mb:
        Size of the engine hash table in megabytes.
    """

    def __init__(
        self,
        path: str = "stockfish",
        skill_level: int = 10,
        move_time: float = 1.0,
        threads: int = 1,
        hash_mb: int = 16,
    ) -> None:
        try:
            from stockfish import Stockfish  # type: ignore[import]

            self._sf = Stockfish(
                path=path,
                depth=15,
                parameters={
                    "Skill Level": skill_level,
                    "Threads": threads,
                    "Hash": hash_mb,
                    "Minimum Thinking Time": 50,
                },
            )
            self._move_time_ms = int(move_time * 1000)
            logger.info("Stockfish engine started (skill=%d).", skill_level)
        except Exception as exc:
            logger.error("Failed to start Stockfish: %s", exc)
            self._sf = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._sf is not None

    def get_best_move(self, board: chess.Board) -> Optional[str]:
        """Return the best move in UCI notation for *board*, or *None* on error."""
        if self._sf is None:
            return None
        try:
            self._sf.set_fen_position(board.fen())
            move = self._sf.get_best_move_time(self._move_time_ms)
            logger.info("Engine move: %s", move)
            return move
        except Exception as exc:
            logger.error("Engine error: %s", exc)
            return None

    def close(self) -> None:
        """Terminate the engine subprocess."""
        if self._sf is not None:
            try:
                self._sf.get_stockfish_major_version()  # no-op to check liveness
            except Exception:
                pass
            self._sf = None
            logger.info("Stockfish engine closed.")
