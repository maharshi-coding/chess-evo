"""Helper functions for the chess vision system."""
from typing import Tuple, List
import numpy as np


# Chess square names
FILES = 'abcdefgh'
RANKS = '12345678'

# Square name to index mapping
SQUARE_TO_INDEX = {
    f"{f}{r}": (7 - int(r) + 1, FILES.index(f))
    for f in FILES for r in RANKS
}

# Index to square name mapping  
INDEX_TO_SQUARE = {v: k for k, v in SQUARE_TO_INDEX.items()}


def square_to_index(square: str) -> Tuple[int, int]:
    """Convert square name (e.g., 'e4') to board index (row, col)."""
    if len(square) != 2:
        raise ValueError(f"Invalid square name: {square}")
    file, rank = square[0].lower(), square[1]
    if file not in FILES or rank not in RANKS:
        raise ValueError(f"Invalid square name: {square}")
    row = 8 - int(rank)  # a8 is (0, 0)
    col = FILES.index(file)
    return (row, col)


def index_to_square(row: int, col: int) -> str:
    """Convert board index (row, col) to square name (e.g., 'e4')."""
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"Invalid index: ({row}, {col})")
    file = FILES[col]
    rank = str(8 - row)
    return f"{file}{rank}"


def is_light_square(square: str) -> bool:
    """Check if a square is a light square on the board."""
    row, col = square_to_index(square)
    return (row + col) % 2 == 0


def uci_to_squares(uci_move: str) -> Tuple[str, str]:
    """Convert UCI move (e.g., 'e2e4') to (from_square, to_square)."""
    if len(uci_move) < 4:
        raise ValueError(f"Invalid UCI move: {uci_move}")
    return uci_move[:2], uci_move[2:4]


def squares_to_uci(from_sq: str, to_sq: str, promotion: str = '') -> str:
    """Convert squares to UCI move format."""
    return f"{from_sq}{to_sq}{promotion}"


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum and diff to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]   # Top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]   # Bottom-left has largest diff
    
    return rect


def get_all_squares() -> List[str]:
    """Get list of all 64 squares in order (a8, b8, ..., h1)."""
    squares = []
    for rank in '87654321':
        for file in FILES:
            squares.append(f"{file}{rank}")
    return squares
