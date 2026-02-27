"""Piece detection using color analysis."""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from enum import IntEnum

from src.config import config
from src.utils.logging_setup import get_logger
from src.utils.helpers import get_all_squares, is_light_square

logger = get_logger(__name__)


class SquareState(IntEnum):
    """State of a square on the board."""
    EMPTY = 0
    WHITE_PIECE = 1
    BLACK_PIECE = -1
    UNKNOWN = 99


class PieceDetector:
    """Detects pieces on squares using color-based analysis."""
    
    def __init__(self):
        """Initialize the piece detector."""
        piece_config = config.vision.get('piece_detection', {})
        self.method = piece_config.get('method', 'color')
        self.empty_threshold = piece_config.get('empty_threshold', 0.3)
        self.white_threshold = piece_config.get('white_threshold', 0.6)
        self.sample_size = piece_config.get('sample_size', 30)
        
        # Reference colors for calibration
        self._empty_light_ref: Optional[np.ndarray] = None
        self._empty_dark_ref: Optional[np.ndarray] = None
        self._white_piece_ref: Optional[np.ndarray] = None
        self._black_piece_ref: Optional[np.ndarray] = None
        
        # Detection thresholds (can be tuned)
        self._piece_presence_threshold = 15.0  # Min color difference for piece
        self._white_black_threshold = 128.0    # Grayscale threshold
    
    def classify_square(self, square_img: np.ndarray, square_name: str) -> SquareState:
        """Classify a single square as empty, white piece, or black piece.
        
        Args:
            square_img: Image of the square (BGR).
            square_name: Name of the square (e.g., 'e4') for calibration reference.
            
        Returns:
            SquareState indicating the square's content.
        """
        if square_img is None or square_img.size == 0:
            return SquareState.UNKNOWN
        
        # Get center region to avoid edge artifacts
        h, w = square_img.shape[:2]
        margin = max(h, w) // 4
        center = square_img[margin:h-margin, margin:w-margin]
        
        if center.size == 0:
            center = square_img
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean and standard deviation
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Get the expected empty square color
        is_light = is_light_square(square_name)
        
        # Simple heuristic based on variance and brightness
        # Empty squares have low variance (uniform color)
        # Squares with pieces have higher variance (piece + square visible)
        
        if std_val < self._piece_presence_threshold:
            # Low variance - likely empty
            return SquareState.EMPTY
        
        # Has piece - determine color based on central brightness
        # Sample the very center where the piece is most visible
        center_small = gray[h//3:2*h//3, w//3:2*w//3] if h > 10 and w > 10 else gray
        center_mean = np.mean(center_small)
        
        if center_mean > self._white_black_threshold:
            return SquareState.WHITE_PIECE
        else:
            return SquareState.BLACK_PIECE
    
    def classify_board(self, squares: Dict[str, np.ndarray]) -> np.ndarray:
        """Classify all squares on the board.
        
        Args:
            squares: Dictionary mapping square names to images.
            
        Returns:
            8x8 numpy array with SquareState values.
        """
        board = np.zeros((8, 8), dtype=np.int8)
        
        for square, img in squares.items():
            state = self.classify_square(img, square)
            
            # Convert square name to index
            col = ord(square[0]) - ord('a')
            row = 8 - int(square[1])
            
            board[row, col] = int(state)
        
        return board
    
    def calibrate_empty(self, squares: Dict[str, np.ndarray], 
                       empty_squares: list) -> None:
        """Calibrate empty square colors.
        
        Args:
            squares: Dictionary of all square images.
            empty_squares: List of square names known to be empty.
        """
        light_samples = []
        dark_samples = []
        
        for sq in empty_squares:
            if sq in squares:
                sample = self._get_center_color(squares[sq])
                if is_light_square(sq):
                    light_samples.append(sample)
                else:
                    dark_samples.append(sample)
        
        if light_samples:
            self._empty_light_ref = np.mean(light_samples, axis=0)
            logger.info(f"Calibrated light square: {self._empty_light_ref}")
        
        if dark_samples:
            self._empty_dark_ref = np.mean(dark_samples, axis=0)
            logger.info(f"Calibrated dark square: {self._empty_dark_ref}")
    
    def calibrate_pieces(self, squares: Dict[str, np.ndarray],
                        white_squares: list, black_squares: list) -> None:
        """Calibrate piece colors.
        
        Args:
            squares: Dictionary of all square images.
            white_squares: List of squares known to have white pieces.
            black_squares: List of squares known to have black pieces.
        """
        white_samples = []
        black_samples = []
        
        for sq in white_squares:
            if sq in squares:
                white_samples.append(self._get_center_color(squares[sq]))
        
        for sq in black_squares:
            if sq in squares:
                black_samples.append(self._get_center_color(squares[sq]))
        
        if white_samples:
            self._white_piece_ref = np.mean(white_samples, axis=0)
            logger.info(f"Calibrated white piece: {self._white_piece_ref}")
        
        if black_samples:
            self._black_piece_ref = np.mean(black_samples, axis=0)
            logger.info(f"Calibrated black piece: {self._black_piece_ref}")
    
    def _get_center_color(self, square_img: np.ndarray) -> np.ndarray:
        """Get the average color of the center region."""
        h, w = square_img.shape[:2]
        margin = max(h, w) // 4
        center = square_img[margin:h-margin, margin:w-margin]
        return np.mean(center, axis=(0, 1))
    
    def set_thresholds(self, piece_presence: float = None, 
                      white_black: float = None) -> None:
        """Adjust detection thresholds.
        
        Args:
            piece_presence: Threshold for piece presence detection.
            white_black: Threshold for white vs black piece detection.
        """
        if piece_presence is not None:
            self._piece_presence_threshold = piece_presence
            logger.info(f"Piece presence threshold set to {piece_presence}")
        
        if white_black is not None:
            self._white_black_threshold = white_black
            logger.info(f"White/black threshold set to {white_black}")
    
    def get_detection_debug(self, square_img: np.ndarray, 
                           square_name: str) -> Dict:
        """Get debug information for square detection.
        
        Args:
            square_img: Image of the square.
            square_name: Square name.
            
        Returns:
            Dictionary with debug metrics.
        """
        if square_img is None or square_img.size == 0:
            return {'error': 'Invalid image'}
        
        h, w = square_img.shape[:2]
        margin = max(h, w) // 4
        center = square_img[margin:h-margin, margin:w-margin]
        
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        
        return {
            'square': square_name,
            'is_light': is_light_square(square_name),
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
            'center_color': self._get_center_color(square_img).tolist(),
            'classification': self.classify_square(square_img, square_name).name
        }
