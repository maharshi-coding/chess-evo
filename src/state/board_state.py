"""Board state representation."""
import chess
import numpy as np
from typing import Optional, Dict, List, Tuple
from copy import deepcopy

from src.utils.logging_setup import get_logger
from src.utils.helpers import square_to_index, index_to_square, get_all_squares
from src.vision.piece_detector import SquareState

logger = get_logger(__name__)


# Piece symbols for display
PIECE_SYMBOLS = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K'
}


class BoardState:
    """Represents the current state of the chess board."""
    
    def __init__(self, fen: Optional[str] = None):
        """Initialize board state.
        
        Args:
            fen: Optional FEN string. If None, uses starting position.
        """
        if fen is None:
            self._board = chess.Board()
        else:
            self._board = chess.Board(fen)
        
        # Detection state (what the camera sees)
        self._detection_grid: Optional[np.ndarray] = None
        
        # Stability tracking
        self._stable_frames = 0
        self._last_detection: Optional[np.ndarray] = None
    
    @classmethod
    def from_fen(cls, fen: str) -> 'BoardState':
        """Create BoardState from FEN string."""
        return cls(fen)
    
    @classmethod
    def starting_position(cls) -> 'BoardState':
        """Create BoardState with starting position."""
        return cls()
    
    @property
    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()
    
    @property
    def board(self) -> chess.Board:
        """Get the underlying python-chess Board object."""
        return self._board
    
    def get_piece_at(self, square: str) -> Optional[str]:
        """Get the piece at a square.
        
        Args:
            square: Square name (e.g., 'e4').
            
        Returns:
            Piece symbol (e.g., 'P', 'n') or None if empty.
        """
        sq = chess.parse_square(square)
        piece = self._board.piece_at(sq)
        return piece.symbol() if piece else None
    
    def set_piece_at(self, square: str, piece: Optional[str]) -> None:
        """Set a piece at a square.
        
        Args:
            square: Square name (e.g., 'e4').
            piece: Piece symbol (e.g., 'P', 'n') or None to clear.
        """
        sq = chess.parse_square(square)
        if piece is None:
            self._board.remove_piece_at(sq)
        else:
            self._board.set_piece_at(sq, chess.Piece.from_symbol(piece))
    
    def copy(self) -> 'BoardState':
        """Create a copy of this board state."""
        new_state = BoardState.__new__(BoardState)
        new_state._board = self._board.copy()
        new_state._detection_grid = self._detection_grid.copy() if self._detection_grid is not None else None
        new_state._stable_frames = self._stable_frames
        new_state._last_detection = self._last_detection.copy() if self._last_detection is not None else None
        return new_state
    
    def update_from_detection(self, detection_grid: np.ndarray, 
                             stability_threshold: int = 3) -> bool:
        """Update state from vision detection.
        
        Args:
            detection_grid: 8x8 array from piece detector.
            stability_threshold: Number of consistent frames required.
            
        Returns:
            True if state changed (move detected), False otherwise.
        """
        if self._last_detection is not None and np.array_equal(detection_grid, self._last_detection):
            self._stable_frames += 1
        else:
            self._stable_frames = 1
            self._last_detection = detection_grid.copy()
        
        if self._stable_frames >= stability_threshold:
            if self._detection_grid is None or not np.array_equal(detection_grid, self._detection_grid):
                self._detection_grid = detection_grid.copy()
                return True
        
        return False
    
    def get_occupied_squares(self, color: Optional[chess.Color] = None) -> List[str]:
        """Get list of occupied squares.
        
        Args:
            color: Optional color filter (chess.WHITE or chess.BLACK).
            
        Returns:
            List of square names.
        """
        squares = []
        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece is not None:
                if color is None or piece.color == color:
                    squares.append(chess.square_name(sq))
        return squares
    
    def get_empty_squares(self) -> List[str]:
        """Get list of empty squares."""
        squares = []
        for sq in chess.SQUARES:
            if self._board.piece_at(sq) is None:
                squares.append(chess.square_name(sq))
        return squares
    
    def make_move(self, move: str) -> bool:
        """Make a move on the board.
        
        Args:
            move: Move in UCI format (e.g., 'e2e4').
            
        Returns:
            True if move was legal and made, False otherwise.
        """
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move in self._board.legal_moves:
                self._board.push(chess_move)
                return True
            return False
        except (ValueError, chess.InvalidMoveError):
            return False
    
    def unmake_move(self) -> Optional[str]:
        """Unmake the last move.
        
        Returns:
            The move that was undone, or None if no moves to undo.
        """
        if len(self._board.move_stack) == 0:
            return None
        move = self._board.pop()
        return move.uci()
    
    def is_check(self) -> bool:
        """Check if current player is in check."""
        return self._board.is_check()
    
    def is_checkmate(self) -> bool:
        """Check if current player is checkmated."""
        return self._board.is_checkmate()
    
    def is_stalemate(self) -> bool:
        """Check if game is stalemate."""
        return self._board.is_stalemate()
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self._board.is_game_over()
    
    @property
    def turn(self) -> chess.Color:
        """Get whose turn it is."""
        return self._board.turn
    
    @property
    def turn_str(self) -> str:
        """Get turn as string ('white' or 'black')."""
        return 'white' if self._board.turn == chess.WHITE else 'black'
    
    @property
    def fullmove_number(self) -> int:
        """Get the full move number."""
        return self._board.fullmove_number
    
    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in UCI format."""
        return [move.uci() for move in self._board.legal_moves]
    
    def to_grid(self) -> np.ndarray:
        """Convert board to 8x8 grid representation.
        
        Returns:
            8x8 array where:
            - Positive values are white pieces (1-6 for P,N,B,R,Q,K)
            - Negative values are black pieces (-1 to -6)
            - 0 is empty
        """
        grid = np.zeros((8, 8), dtype=np.int8)
        
        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece is not None:
                row = 7 - chess.square_rank(sq)
                col = chess.square_file(sq)
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                grid[row, col] = value
        
        return grid
    
    def to_occupancy_grid(self) -> np.ndarray:
        """Convert board to occupancy grid (like piece detector output).
        
        Returns:
            8x8 array where:
            - 1 is white piece
            - -1 is black piece
            - 0 is empty
        """
        grid = np.zeros((8, 8), dtype=np.int8)
        
        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece is not None:
                row = 7 - chess.square_rank(sq)
                col = chess.square_file(sq)
                grid[row, col] = 1 if piece.color == chess.WHITE else -1
        
        return grid
    
    def __str__(self) -> str:
        """String representation of the board."""
        return str(self._board)
    
    def __repr__(self) -> str:
        return f"BoardState(fen='{self.fen}')"
