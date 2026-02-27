"""Move detection from successive board states.

Performance design
------------------
* Candidate moves are derived by computing the *diff* between two
  ``BoardState`` objects (a single ``np.where`` call) and checking only the
  changed squares (≤ 4) against the legal move list.
* ``chess.Board.legal_moves`` is a generator; we only iterate as far as needed
  and exit early once a matching move is found.
* Repeated state snapshots are stored as **references** to lightweight
  numpy-backed ``BoardState`` objects (not full board deep-copies).
* A configurable ``confirmation_frames`` counter prevents spurious move
  detections caused by sensor noise – a candidate must persist across N
  consecutive frames before being accepted.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np

from src.state.board_state import BoardState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedMove:
    from_square: str
    to_square: str
    uci: str
    is_castling: bool = False


class MoveDetector:
    """Detect chess moves by comparing successive board occupancy states.

    Parameters
    ----------
    confirmation_frames:
        How many consecutive frames must show the same change before the
        move is reported as confirmed.
    """

    def __init__(self, confirmation_frames: int = 5) -> None:
        self._confirm_n = confirmation_frames
        # Ring buffer holding the last N occupancy grids for stability check.
        self._recent: deque[np.ndarray] = deque(maxlen=confirmation_frames)
        self._last_accepted: Optional[BoardState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_state: BoardState) -> None:
        """Initialise (or reset) with the given starting board state."""
        self._recent.clear()
        self._last_accepted = initial_state.copy()

    def push(self, current: BoardState, chess_board: chess.Board) -> Optional[chess.Move]:
        """Feed the latest observed board state.

        Returns the detected legal ``chess.Move`` when a stable change has
        been confirmed, or *None* otherwise.
        """
        self._recent.append(current.grid.copy())

        if len(self._recent) < self._confirm_n:
            return None

        # All frames in the window must agree (stable state).
        reference = self._recent[0]
        if not all(np.array_equal(reference, g) for g in self._recent):
            return None  # Still fluctuating – wait for stability.

        if self._last_accepted is None:
            return None

        # Compute changed squares (≤ 4 for any legal move).
        stable_state = BoardState()
        stable_state.update(reference)
        changed_warp = self._last_accepted.diff(stable_state)

        if changed_warp.size == 0:
            return None  # No change.

        move = self._infer_move(changed_warp, self._last_accepted, stable_state, chess_board)
        if move is not None:
            logger.info("Move confirmed: %s", move.uci())
            self._last_accepted = stable_state.copy()
            self._recent.clear()
        return move

    def get_changed_squares(self, prev: np.ndarray, curr: np.ndarray) -> list[str]:
        changed: list[str] = []
        for row, col in np.argwhere(prev != curr):
            file_char = chr(ord("a") + int(col))
            rank_char = str(8 - int(row))
            changed.append(f"{file_char}{rank_char}")
        return changed

    def detect_move(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        chess_board: chess.Board,
    ) -> Optional[DetectedMove]:
        if prev.shape != (8, 8) or curr.shape != (8, 8):
            return None

        def _board_to_occ(board: chess.Board) -> np.ndarray:
            occ = np.zeros((8, 8), dtype=np.int8)
            for sq, piece in board.piece_map().items():
                row = 7 - chess.square_rank(sq)
                col = chess.square_file(sq)
                occ[row, col] = 1 if piece.color == chess.WHITE else -1
            return occ

        for legal_move in chess_board.legal_moves:
            after_board = chess_board.copy(stack=False)
            after_board.push(legal_move)
            if np.array_equal(_board_to_occ(after_board), curr):
                return DetectedMove(
                    from_square=chess.square_name(legal_move.from_square),
                    to_square=chess.square_name(legal_move.to_square),
                    uci=legal_move.uci(),
                    is_castling=chess_board.is_castling(legal_move),
                )

        before = BoardState()
        before.update(prev.reshape(64).astype(np.int8))
        after = BoardState()
        after.update(curr.reshape(64).astype(np.int8))

        changed = before.diff(after)
        move = self._infer_move(changed, before, after, chess_board)
        if move is None:
            return None

        return DetectedMove(
            from_square=chess.square_name(move.from_square),
            to_square=chess.square_name(move.to_square),
            uci=move.uci(),
            is_castling=chess_board.is_castling(move),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_move(
        changed_warp: np.ndarray,
        before: BoardState,
        after: BoardState,
        chess_board: chess.Board,
    ) -> Optional[chess.Move]:
        """Attempt to identify a legal chess move from the set of changed squares.

        Strategy:
        1. Convert warp indices → python-chess squares.
        2. Find which square *lost* a piece (from-square) and which *gained*
           one (to-square).
        3. Iterate legal moves with matching from/to squares – exit early on
           first match.
        """
        changed_chess = [BoardState.warp_to_chess_sq(int(w)) for w in changed_warp]

        from_sq: Optional[int] = None
        to_sq: Optional[int] = None

        for sq in changed_chess:
            warp_idx = BoardState.chess_sq_to_warp(sq)
            occ_before = int(before.grid[warp_idx])
            occ_after = int(after.grid[warp_idx])

            if occ_before != 0 and occ_after == 0:
                from_sq = sq  # Piece lifted.
            elif occ_after != 0:
                to_sq = sq    # Piece placed (capture or move-to).

        if from_sq is None or to_sq is None:
            logger.debug(
                "Cannot infer move from changed squares: %s", changed_chess
            )
            return None

        # Early-exit search through legal moves.
        for move in chess_board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                return move

        logger.debug("No legal move found for %s→%s.", chess.square_name(from_sq), chess.square_name(to_sq))
        return None
