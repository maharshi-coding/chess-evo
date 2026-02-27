"""Chessboard corner and square detection.

Performance design
------------------
* Board corners are expensive to compute (contour analysis, perspective
  transform), so they are **cached** once detected and only recomputed on
  explicit ``reset()``.
* The 64 square ROI bounding boxes are pre-computed as a (64, 4) numpy array
  the first time the board is warped, enabling vectorised slicing later.
* The warp transform matrix is stored so applying it to new frames is a single
  ``cv2.warpPerspective`` call rather than recomputing it each time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Board is warped to this resolution for downstream processing.
WARP_SIZE = 512
SQUARE_PX = WARP_SIZE // 8  # 64 px per square


@dataclass
class BoardDetectionResult:
    found: bool
    corners: Optional[np.ndarray] = None
    confidence: float = 0.0
    warped: Optional[np.ndarray] = None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the four corner points in (TL, TR, BR, BL) order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def _square_rois() -> np.ndarray:
    """Return a (64, 4) array of (x1, y1, x2, y2) ROIs for each square.

    Indexed row-major from a8 (index 0) to h1 (index 63), consistent with
    python-chess ``chess.A8 == 56`` when converted by ``_sq_index()``.
    Cached so it is built only once per process.
    """
    rois = np.empty((64, 4), dtype=np.int32)
    for rank in range(8):
        for file in range(8):
            sq = rank * 8 + file
            x1 = file * SQUARE_PX
            y1 = rank * SQUARE_PX
            rois[sq] = [x1, y1, x1 + SQUARE_PX, y1 + SQUARE_PX]
    return rois


class BoardDetector:
    """Detect and warp the chessboard from a camera frame.

    Parameters
    ----------
    min_area_fraction, max_area_fraction:
        The board contour must cover between these fractions of the full
        frame area to be accepted.
    """

    def __init__(
        self,
        min_area_fraction: float = 0.10,
        max_area_fraction: float = 0.95,
    ) -> None:
        self._min_frac = min_area_fraction
        self._max_frac = max_area_fraction

        # Cached state – reset when calibrating.
        self._corners: Optional[np.ndarray] = None   # shape (4, 2), float32
        self._transform: Optional[np.ndarray] = None  # 3x3 homography matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Invalidate cached corners, forcing re-detection on next call."""
        self._corners = None
        self._transform = None
        logger.info("BoardDetector cache cleared.")

    def reset_calibration(self) -> None:
        self.reset()

    def calibrate(self, corners: np.ndarray) -> None:
        ordered = _order_corners(corners)
        self._corners = ordered
        self._transform = self._build_transform(ordered)

    @property
    def calibrated_corners(self) -> Optional[np.ndarray]:
        return None if self._corners is None else self._corners.copy()

    @property
    def is_calibrated(self) -> bool:
        return self._corners is not None

    def detect(self, frame: np.ndarray) -> BoardDetectionResult:
        """Return board detection result including warped bird's-eye image.

        Uses cached corners when available; performs detection otherwise.
        Returns ``BoardDetectionResult(found=False)`` when the board cannot be found.
        """
        if self._transform is None:
            corners = self._find_corners(frame)
            if corners is None:
                return BoardDetectionResult(found=False)
            self._corners = corners
            self._transform = self._build_transform(corners)

        warped: np.ndarray = cv2.warpPerspective(
            frame,
            self._transform,
            (WARP_SIZE, WARP_SIZE),
            flags=cv2.INTER_LINEAR,
        )
        return BoardDetectionResult(
            found=True,
            corners=None if self._corners is None else self._corners.copy(),
            confidence=1.0,
            warped=warped,
        )

    def get_square_image(self, warped: np.ndarray, sq: int) -> np.ndarray:
        """Return the warped image crop for chess square *sq* (0-63)."""
        rois = _square_rois()
        x1, y1, x2, y2 = rois[sq]
        return warped[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Corner detection helpers
    # ------------------------------------------------------------------

    def _find_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Locate the four chessboard corners using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold handles varying illumination.
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.debug("No contours found during board detection.")
            return None

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = self._min_frac * frame_area
        max_area = self._max_frac * frame_area

        best: Optional[np.ndarray] = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best = approx
                best_area = area

        if best is None:
            logger.debug("No quadrilateral board contour found.")
            return None

        corners = _order_corners(best)
        logger.info("Board corners detected (area=%.0f px²).", best_area)
        return corners

    @staticmethod
    def _build_transform(corners: np.ndarray) -> np.ndarray:
        """Compute the perspective transform from *corners* to a square warp."""
        dst = np.array(
            [
                [0, 0],
                [WARP_SIZE - 1, 0],
                [WARP_SIZE - 1, WARP_SIZE - 1],
                [0, WARP_SIZE - 1],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(corners, dst)
