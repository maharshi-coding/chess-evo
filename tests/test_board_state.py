"""Tests for BoardState – the core numpy-backed occupancy grid."""

from __future__ import annotations

import numpy as np
import pytest
import chess

from src.state.board_state import BoardState, _PY_TO_WARP, _WARP_TO_PY


class TestBoardStateInit:
    def test_initial_grid_is_all_zeros(self):
        bs = BoardState()
        assert bs.grid.shape == (64,)
        assert bs.grid.dtype == np.int8
        assert np.all(bs.grid == 0)


class TestBoardStateUpdate:
    def test_update_replaces_grid(self):
        bs = BoardState()
        occ = np.ones(64, dtype=np.int8)
        bs.update(occ)
        assert np.array_equal(bs.grid, occ)

    def test_update_wrong_shape_raises(self):
        bs = BoardState()
        with pytest.raises(ValueError):
            bs.update(np.zeros(32, dtype=np.int8))

    def test_update_is_independent_copy(self):
        """Mutating the source array after update must not affect stored state."""
        bs = BoardState()
        occ = np.zeros(64, dtype=np.int8)
        bs.update(occ)
        occ[0] = 1  # mutate source
        assert bs.grid[0] == 0  # stored value unchanged


class TestBoardStateDiff:
    def test_diff_returns_indices_of_changed_squares(self):
        bs1 = BoardState()
        bs2 = BoardState()
        occ = np.zeros(64, dtype=np.int8)
        occ[10] = 1
        occ[20] = -1
        bs2.update(occ)
        changed = bs1.diff(bs2)
        assert set(changed.tolist()) == {10, 20}

    def test_diff_empty_when_equal(self):
        bs1 = BoardState()
        bs2 = BoardState()
        assert bs1.diff(bs2).size == 0

    def test_diff_all_squares_when_fully_different(self):
        bs1 = BoardState()
        occ = np.ones(64, dtype=np.int8)
        bs2 = BoardState()
        bs2.update(occ)
        assert bs1.diff(bs2).size == 64


class TestBoardStateEquals:
    def test_equals_true_for_identical_grids(self):
        bs1 = BoardState()
        bs2 = BoardState()
        assert bs1.equals(bs2)

    def test_equals_false_for_different_grids(self):
        bs1 = BoardState()
        bs2 = BoardState()
        occ = np.zeros(64, dtype=np.int8)
        occ[5] = 1
        bs2.update(occ)
        assert not bs1.equals(bs2)


class TestBoardStateCopy:
    def test_copy_is_independent(self):
        bs1 = BoardState()
        occ = np.zeros(64, dtype=np.int8)
        occ[3] = 1
        bs1.update(occ)
        bs2 = bs1.copy()
        # Mutate original after copy.
        occ2 = bs1.grid.copy()
        occ2[3] = 0
        bs1.update(occ2)
        assert bs2.grid[3] == 1  # copy unaffected


class TestSquareIndexConversions:
    def test_warp_to_chess_round_trip(self):
        for warp_idx in range(64):
            chess_sq = BoardState.warp_to_chess_sq(warp_idx)
            assert BoardState.chess_sq_to_warp(chess_sq) == warp_idx

    def test_chess_to_warp_round_trip(self):
        for chess_sq in chess.SQUARES:
            warp_idx = BoardState.chess_sq_to_warp(chess_sq)
            assert BoardState.warp_to_chess_sq(warp_idx) == chess_sq

    def test_a8_is_warp_index_0(self):
        """The top-left of the warped image corresponds to a8."""
        assert BoardState.chess_sq_to_warp(chess.A8) == 0

    def test_h1_is_warp_index_63(self):
        """The bottom-right of the warped image corresponds to h1."""
        assert BoardState.chess_sq_to_warp(chess.H1) == 63
