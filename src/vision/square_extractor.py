"""Square extraction from board image."""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_logger
from src.utils.helpers import get_all_squares, index_to_square

logger = get_logger(__name__)


class SquareExtractor:
    """Extracts individual squares from a transformed board image."""
    
    def __init__(self, board_size: int = 640):
        """Initialize the square extractor.
        
        Args:
            board_size: Size of the input board image in pixels.
        """
        self.board_size = board_size
        self.square_size = board_size // 8
        
        # Precompute square boundaries
        self._square_bounds: Dict[str, Tuple[int, int, int, int]] = {}
        self._compute_square_bounds()
    
    def _compute_square_bounds(self) -> None:
        """Precompute the bounding boxes for all squares."""
        for row in range(8):
            for col in range(8):
                square = index_to_square(row, col)
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self._square_bounds[square] = (x1, y1, x2, y2)
    
    def extract_all(self, board_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all 64 squares from the board image.
        
        Args:
            board_image: Top-down view of the board (square image).
            
        Returns:
            Dictionary mapping square names (e.g., 'e4') to image patches.
        """
        if board_image is None:
            return {}
        
        # Resize if necessary
        if board_image.shape[0] != self.board_size or board_image.shape[1] != self.board_size:
            board_image = cv2.resize(board_image, (self.board_size, self.board_size))
        
        squares = {}
        for square, (x1, y1, x2, y2) in self._square_bounds.items():
            squares[square] = board_image[y1:y2, x1:x2].copy()
        
        return squares
    
    def extract_square(self, board_image: np.ndarray, square: str) -> Optional[np.ndarray]:
        """Extract a single square from the board image.
        
        Args:
            board_image: Top-down view of the board.
            square: Square name (e.g., 'e4').
            
        Returns:
            Image patch of the square, or None if invalid.
        """
        if square not in self._square_bounds:
            logger.warning(f"Invalid square name: {square}")
            return None
        
        # Resize if necessary
        if board_image.shape[0] != self.board_size or board_image.shape[1] != self.board_size:
            board_image = cv2.resize(board_image, (self.board_size, self.board_size))
        
        x1, y1, x2, y2 = self._square_bounds[square]
        return board_image[y1:y2, x1:x2].copy()
    
    def get_square_center(self, square: str) -> Optional[Tuple[int, int]]:
        """Get the center coordinates of a square in the board image.
        
        Args:
            square: Square name (e.g., 'e4').
            
        Returns:
            (x, y) center coordinates, or None if invalid square.
        """
        if square not in self._square_bounds:
            return None
        
        x1, y1, x2, y2 = self._square_bounds[square]
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_square_bounds(self, square: str) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box of a square.
        
        Args:
            square: Square name (e.g., 'e4').
            
        Returns:
            (x1, y1, x2, y2) bounds, or None if invalid square.
        """
        return self._square_bounds.get(square)
    
    def create_debug_image(self, board_image: np.ndarray, 
                          highlight_squares: Optional[List[str]] = None) -> np.ndarray:
        """Create a debug image showing square boundaries.
        
        Args:
            board_image: Top-down view of the board.
            highlight_squares: Optional list of squares to highlight.
            
        Returns:
            Debug image with grid lines and labels.
        """
        debug = board_image.copy()
        
        # Resize if necessary
        if debug.shape[0] != self.board_size or debug.shape[1] != self.board_size:
            debug = cv2.resize(debug, (self.board_size, self.board_size))
        
        # Draw grid lines
        for i in range(9):
            pos = i * self.square_size
            cv2.line(debug, (pos, 0), (pos, self.board_size), (0, 255, 0), 1)
            cv2.line(debug, (0, pos), (self.board_size, pos), (0, 255, 0), 1)
        
        # Highlight specific squares
        if highlight_squares:
            for square in highlight_squares:
                if square in self._square_bounds:
                    x1, y1, x2, y2 = self._square_bounds[square]
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add square labels
        for square, (x1, y1, x2, y2) in self._square_bounds.items():
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(debug, square, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug
    
    def point_to_square(self, x: int, y: int) -> Optional[str]:
        """Convert board coordinates to square name.
        
        Args:
            x: X coordinate in board image.
            y: Y coordinate in board image.
            
        Returns:
            Square name (e.g., 'e4'), or None if outside board.
        """
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return None
        
        col = x // self.square_size
        row = y // self.square_size
        
        # Clamp to valid range
        col = min(7, max(0, col))
        row = min(7, max(0, row))
        
        return index_to_square(row, col)
