"""Perspective transformation for board images."""
import cv2
import numpy as np
from typing import Optional, Tuple

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class PerspectiveTransformer:
    """Handles perspective transformation to get a top-down view of the board."""
    
    def __init__(self, output_size: int = 640):
        """Initialize the transformer.
        
        Args:
            output_size: Size of the output square image in pixels.
        """
        self.output_size = output_size
        self._transform_matrix: Optional[np.ndarray] = None
        self._inverse_matrix: Optional[np.ndarray] = None
        
        # Destination points for the top-down view
        self._dst_points = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)
    
    def compute_transform(self, corners: np.ndarray) -> np.ndarray:
        """Compute the perspective transform matrix.
        
        Args:
            corners: 4 corner points of the board (TL, TR, BR, BL).
            
        Returns:
            The 3x3 transformation matrix.
        """
        if corners.shape != (4, 2):
            raise ValueError("Corners must be shape (4, 2)")
        
        self._transform_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            self._dst_points
        )
        self._inverse_matrix = cv2.getPerspectiveTransform(
            self._dst_points,
            corners.astype(np.float32)
        )
        
        return self._transform_matrix
    
    def transform(self, frame: np.ndarray, corners: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Transform frame to get top-down view of the board.
        
        Args:
            frame: Input BGR image.
            corners: Optional corner points. If None, uses cached transform.
            
        Returns:
            Transformed image of the board, or None if transform not available.
        """
        if corners is not None:
            self.compute_transform(corners)
        
        if self._transform_matrix is None:
            logger.warning("Transform matrix not computed")
            return None
        
        warped = cv2.warpPerspective(
            frame,
            self._transform_matrix,
            (self.output_size, self.output_size)
        )
        
        return warped
    
    def inverse_transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform a point from board coordinates back to frame coordinates.
        
        Args:
            point: Point in board coordinates (x, y).
            
        Returns:
            Point in original frame coordinates, or None if transform not available.
        """
        if self._inverse_matrix is None:
            return None
        
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._inverse_matrix)
        return tuple(transformed[0, 0])
    
    def transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform a point from frame coordinates to board coordinates.
        
        Args:
            point: Point in frame coordinates (x, y).
            
        Returns:
            Point in board coordinates, or None if transform not available.
        """
        if self._transform_matrix is None:
            return None
        
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._transform_matrix)
        return tuple(transformed[0, 0])
    
    @property
    def is_ready(self) -> bool:
        """Check if transform is ready to use."""
        return self._transform_matrix is not None
    
    @property
    def transform_matrix(self) -> Optional[np.ndarray]:
        """Get the current transform matrix."""
        return self._transform_matrix.copy() if self._transform_matrix is not None else None
