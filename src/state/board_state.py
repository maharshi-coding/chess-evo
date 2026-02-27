"""Board state representation and occupancy grid management.

Performance design
------------------
* Internal state is a compact ``numpy.ndarray`` of shape (64,) with dtype
  ``int8``.  Comparing two states is a single ``np.array_equal`` call (O(64)
  C-level loop) rather than 64 Python comparisons.
* ``diff()`` uses ``np.where(a != b)`` which runs entirely in C and returns a
  pre-allocated index array.
* Square-name ↔ index mappings are module-level tuples so they are built once
  at import time with zero per-call overhead.
"""

from __future__ import annotations

import logging
from typing import Optional

import chess
import numpy as np

logger = logging.getLogger(__name__)

# Map python-chess square integers to our row-major warped-board indices.
# python-chess: A1=0 … H8=63, rank-0 is rank 1 (white's back rank).
# Our warp: row-0 is the top of the warped image (black's back rank = rank 8).
# So warp_index = (7 - rank) * 8 + file
_PY_TO_WARP: tuple[int, ...] = tuple(
    (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)
    for sq in chess.SQUARES
)
_WARP_TO_PY: tuple[int, ...] = tuple(
    chess.square(warp_idx % 8, 7 - warp_idx // 8) for warp_idx in range(64)
)


class BoardState:
    """Stores and compares chess board occupancy grids.

    Occupancy values: ``1`` = white piece, ``-1`` = black piece, ``0`` = empty.
    Indices follow the warp-image row-major order (a8=0, h1=63).
    """

    def __init__(self) -> None:
        self._grid: np.ndarray = np.zeros(64, dtype=np.int8)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Read-only view of the occupancy grid."""
        return self._grid.view()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, occupancy: np.ndarray) -> None:
        """Replace the internal grid with *occupancy* (shape (64,) int8)."""
        if occupancy.shape != (64,):
            raise ValueError(f"Expected shape (64,), got {occupancy.shape}")
        np.copyto(self._grid, occupancy)

    def diff(self, other: "BoardState") -> np.ndarray:
        """Return warp indices where *self* and *other* differ.

        Uses ``np.where`` for a single C-level scan – O(64) with no Python loop.
        """
        (changed,) = np.where(self._grid != other._grid)
        return changed

    def equals(self, other: "BoardState") -> bool:
        """True when both grids are identical."""
        return bool(np.array_equal(self._grid, other._grid))

    def copy(self) -> "BoardState":
        bs = BoardState()
        bs._grid = self._grid.copy()
        return bs

    @staticmethod
    def warp_to_chess_sq(warp_idx: int) -> int:
        """Convert a warp-grid index to a python-chess square integer."""
        return _WARP_TO_PY[warp_idx]

    @staticmethod
    def chess_sq_to_warp(chess_sq: int) -> int:
        """Convert a python-chess square integer to a warp-grid index."""
        return _PY_TO_WARP[chess_sq]

    def to_chess_board(self) -> chess.Board:
        """Construct a chess.Board from the current occupancy grid.

        **Note**: Occupancy alone cannot reconstruct piece *types* – this
        produces a board with generic pawns and is useful only for debugging.
        """
        board = chess.Board(fen=None)
        for warp_idx, occ in enumerate(self._grid):
            if occ == 0:
                continue
            sq = _WARP_TO_PY[warp_idx]
            color = chess.WHITE if occ > 0 else chess.BLACK
            board.set_piece_at(sq, chess.Piece(chess.PAWN, color))
        return board

    def __repr__(self) -> str:  # pragma: no cover
        return f"BoardState(non_empty={int(np.count_nonzero(self._grid))})"
