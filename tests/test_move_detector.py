"""Tests for MoveDetector – move inference from board state changes."""

from __future__ import annotations

import numpy as np
import pytest
import chess

from src.state.board_state import BoardState
from src.state.move_detector import MoveDetector


def _make_state(occupied: dict[int, int]) -> BoardState:
    """Build a BoardState from a {warp_index: occupancy} dict."""
    bs = BoardState()
    grid = np.zeros(64, dtype=np.int8)
    for idx, val in occupied.items():
        grid[idx] = val
    bs.update(grid)
    return bs


class TestMoveDetectorConfirmation:
    def test_no_move_before_confirmation_frames(self):
        """Fewer than confirmation_frames pushes must return None."""
        detector = MoveDetector(confirmation_frames=5)
        board = chess.Board()
        initial = _make_state({})
        detector.reset(initial)

        changed = _make_state({1: 1})
        for _ in range(4):
            result = detector.push(changed, board)
            assert result is None

    def test_fluctuating_frames_do_not_confirm(self):
        """Alternating frames must not trigger a false positive."""
        detector = MoveDetector(confirmation_frames=3)
        board = chess.Board()
        initial = _make_state({})
        detector.reset(initial)

        state_a = _make_state({1: 1})
        state_b = _make_state({2: 1})
        for _ in range(6):
            assert detector.push(state_a, board) is None
            assert detector.push(state_b, board) is None


class TestMoveInference:
    def _board_after_e2e4(self) -> tuple[chess.Board, BoardState, BoardState]:
        """Return (board_before_move, state_before, state_after) for e2-e4."""
        board = chess.Board()

        # e2 = chess.E2, e4 = chess.E4
        e2_warp = BoardState.chess_sq_to_warp(chess.E2)
        e4_warp = BoardState.chess_sq_to_warp(chess.E4)

        # Before: white pawn on e2 (warp index for E2)
        before = _make_state({e2_warp: 1})
        # After: pawn moved to e4
        after = _make_state({e4_warp: 1})
        return board, before, after

    def test_infers_e2_e4(self):
        board, before, after = self._board_after_e2e4()
        changed = before.diff(after)
        move = MoveDetector._infer_move(changed, before, after, board)
        assert move is not None
        assert move == chess.Move.from_uci("e2e4")

    def test_no_inference_when_no_from_square(self):
        """If no square lost a piece we cannot determine the move source."""
        board = chess.Board()
        before = _make_state({})
        # Two squares gained pieces but none lost – ambiguous.
        after = _make_state({0: 1, 1: 1})
        changed = before.diff(after)
        move = MoveDetector._infer_move(changed, before, after, board)
        assert move is None

    def test_no_inference_for_illegal_move(self):
        """A physically possible change that has no matching legal move returns None."""
        board = chess.Board()
        # Moving e2 pawn to e6 is not legal from the start position.
        e2_warp = BoardState.chess_sq_to_warp(chess.E2)
        e6_warp = BoardState.chess_sq_to_warp(chess.E6)
        before = _make_state({e2_warp: 1})
        after = _make_state({e6_warp: 1})
        changed = before.diff(after)
        move = MoveDetector._infer_move(changed, before, after, board)
        assert move is None


class TestMoveDetectorReset:
    def test_push_after_reset_uses_new_baseline(self):
        detector = MoveDetector(confirmation_frames=2)
        board = chess.Board()

        initial = _make_state({0: 1})
        detector.reset(initial)

        # Push the same state N times – no move should fire.
        for _ in range(5):
            result = detector.push(initial, board)
            assert result is None
