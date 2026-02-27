"""Piece occupancy detection from a warped board image.

Performance design
------------------
* All 64 squares are processed **in one vectorised pass** using pre-built ROI
  slices instead of a per-square loop.  This avoids Python-level iteration
  over the 64 squares and lets NumPy/OpenCV operate in C on bulk data.
* The HSV conversion and mask operations are applied to the **full warped
  board** once and the results are reused for all 64 squares.
* A small centre crop of each square (inner 50 %) reduces the influence of
  border artefacts introduced by the perspective warp.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.vision.board_detector import SQUARE_PX, WARP_SIZE

logger = logging.getLogger(__name__)

# Keep only the central 50 % of each square to avoid warp-edge artefacts.
_CROP_MARGIN = SQUARE_PX // 4  # 25 % margin on each side


class PieceDetector:
    """Detect white/black piece occupancy on all 64 squares in one pass.

    Parameters
    ----------
    white_hsv_lower, white_hsv_upper:
        HSV bounds for white pieces (H 0-180, S 0-255, V 0-255).
    black_hsv_lower, black_hsv_upper:
        HSV bounds for black pieces.
    min_fill_fraction:
        Minimum fraction of a square's centre crop that must match the piece
        colour for the square to be considered occupied.
    """

    def __init__(
        self,
        white_hsv_lower: tuple[int, int, int] = (0, 0, 180),
        white_hsv_upper: tuple[int, int, int] = (180, 60, 255),
        black_hsv_lower: tuple[int, int, int] = (0, 0, 0),
        black_hsv_upper: tuple[int, int, int] = (180, 255, 80),
        min_fill_fraction: float = 0.10,
    ) -> None:
        self._white_lo = np.array(white_hsv_lower, dtype=np.uint8)
        self._white_hi = np.array(white_hsv_upper, dtype=np.uint8)
        self._black_lo = np.array(black_hsv_lower, dtype=np.uint8)
        self._black_hi = np.array(black_hsv_upper, dtype=np.uint8)
        self._min_fill = min_fill_fraction
        self._crop_area = (SQUARE_PX - 2 * _CROP_MARGIN) ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Return a (64,) int8 array: 1 = white, -1 = black, 0 = empty.

        The entire board is processed in two bulk mask operations (one for
        white, one for black) rather than 64 individual per-square passes.
        """
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, self._white_lo, self._white_hi)
        black_mask = cv2.inRange(hsv, self._black_lo, self._black_hi)

        occupancy = np.zeros(64, dtype=np.int8)

        # Vectorised fill-fraction computation for all squares at once.
        white_fills = self._fill_fractions(white_mask)
        black_fills = self._fill_fractions(black_mask)

        occupancy[white_fills >= self._min_fill] = 1
        # Black overwrites white where both conditions are met (piece contrast).
        occupancy[black_fills >= self._min_fill] = -1

        return occupancy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_fractions(self, mask: np.ndarray) -> np.ndarray:
        """Return the fill fraction for every square's inner crop (vectorised).

        The (WARP_SIZE × WARP_SIZE) mask is reshaped into a
        (8, SQUARE_PX, 8, SQUARE_PX) view, transposed to
        (8, 8, SQUARE_PX, SQUARE_PX), then flattened to (64, SQUARE_PX,
        SQUARE_PX).  The centre crop is extracted with a single slice and
        the per-square non-zero counts are computed with one ``sum`` call —
        no Python loop over the 64 squares.
        """
        m = _CROP_MARGIN
        sq = SQUARE_PX
        # (WARP_SIZE, WARP_SIZE) → (8, sq, 8, sq) → (8, 8, sq, sq) → (64, sq, sq)
        blocks = mask.reshape(8, sq, 8, sq).transpose(0, 2, 1, 3).reshape(64, sq, sq)
        inner = blocks[:, m: sq - m, m: sq - m]          # (64, crop, crop)
        # cv2.inRange produces 0 or 255; treat any non-zero pixel as occupied.
        return (inner.reshape(64, -1) > 0).sum(axis=1).astype(np.float32) / self._crop_area
