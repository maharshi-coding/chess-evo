"""Move validation using chess rules."""
import chess
from typing import List, Optional, Tuple
from enum import Enum

from src.utils.logging_setup import get_logger
from .board_state import BoardState

logger = get_logger(__name__)


class MoveType(Enum):
    """Types of chess moves."""
    NORMAL = "normal"
    CAPTURE = "capture"
    CASTLE_KINGSIDE = "castle_kingside"
    CASTLE_QUEENSIDE = "castle_queenside"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"
    PROMOTION_CAPTURE = "promotion_capture"


class ValidationResult:
    """Result of move validation."""
    
    def __init__(self, is_valid: bool, move: Optional[str] = None,
                 move_type: Optional[MoveType] = None,
                 error: Optional[str] = None):
        self.is_valid = is_valid
        self.move = move  # Canonical UCI format
        self.move_type = move_type
        self.error = error
    
    def __bool__(self) -> bool:
        return self.is_valid


class MoveValidator:
    """Validates chess moves against the current position."""
    
    def __init__(self):
        """Initialize the move validator."""
        pass
    
    def validate(self, move: str, board_state: BoardState) -> ValidationResult:
        """Validate a move against the current board state.
        
        Args:
            move: Move in UCI format (e.g., 'e2e4', 'e7e8q').
            board_state: Current board state.
            
        Returns:
            ValidationResult indicating if move is valid.
        """
        try:
            chess_move = chess.Move.from_uci(move)
        except (ValueError, chess.InvalidMoveError) as e:
            return ValidationResult(
                is_valid=False,
                error=f"Invalid move format: {move}"
            )
        
        board = board_state.board
        
        # Check if move is legal
        if chess_move not in board.legal_moves:
            # Try adding promotion if it's a pawn reaching final rank
            if self._should_promote(chess_move, board):
                # Try with queen promotion by default
                promoted_move = chess.Move(
                    chess_move.from_square, 
                    chess_move.to_square,
                    promotion=chess.QUEEN
                )
                if promoted_move in board.legal_moves:
                    move_type = self._get_move_type(promoted_move, board)
                    return ValidationResult(
                        is_valid=True,
                        move=promoted_move.uci(),
                        move_type=move_type
                    )
            
            # Provide helpful error message
            piece = board.piece_at(chess_move.from_square)
            if piece is None:
                return ValidationResult(
                    is_valid=False,
                    error=f"No piece at {chess.square_name(chess_move.from_square)}"
                )
            
            if piece.color != board.turn:
                expected = "White" if board.turn == chess.WHITE else "Black"
                return ValidationResult(
                    is_valid=False,
                    error=f"It's {expected}'s turn, but piece is {piece.color}"
                )
            
            return ValidationResult(
                is_valid=False,
                error=f"Illegal move: {move}"
            )
        
        move_type = self._get_move_type(chess_move, board)
        
        return ValidationResult(
            is_valid=True,
            move=chess_move.uci(),
            move_type=move_type
        )
    
    def _should_promote(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move should be a promotion."""
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if piece.color == chess.WHITE and to_rank == 7:
                return True
            if piece.color == chess.BLACK and to_rank == 0:
                return True
        return False
    
    def _get_move_type(self, move: chess.Move, board: chess.Board) -> MoveType:
        """Determine the type of move."""
        # Check castling
        if board.is_castling(move):
            if board.is_kingside_castling(move):
                return MoveType.CASTLE_KINGSIDE
            else:
                return MoveType.CASTLE_QUEENSIDE
        
        # Check en passant
        if board.is_en_passant(move):
            return MoveType.EN_PASSANT
        
        # Check promotion
        if move.promotion:
            if board.is_capture(move):
                return MoveType.PROMOTION_CAPTURE
            return MoveType.PROMOTION
        
        # Check capture
        if board.is_capture(move):
            return MoveType.CAPTURE
        
        return MoveType.NORMAL
    
    def get_legal_moves(self, board_state: BoardState) -> List[str]:
        """Get all legal moves in UCI format.
        
        Args:
            board_state: Current board state.
            
        Returns:
            List of legal moves in UCI format.
        """
        return [move.uci() for move in board_state.board.legal_moves]
    
    def get_legal_moves_from(self, board_state: BoardState, 
                            square: str) -> List[str]:
        """Get legal moves from a specific square.
        
        Args:
            board_state: Current board state.
            square: Square name (e.g., 'e2').
            
        Returns:
            List of legal moves from that square in UCI format.
        """
        try:
            from_sq = chess.parse_square(square)
        except ValueError:
            return []
        
        moves = []
        for move in board_state.board.legal_moves:
            if move.from_square == from_sq:
                moves.append(move.uci())
        
        return moves
    
    def is_in_check(self, board_state: BoardState) -> bool:
        """Check if current player is in check."""
        return board_state.board.is_check()
    
    def gives_check(self, move: str, board_state: BoardState) -> bool:
        """Check if a move gives check.
        
        Args:
            move: Move in UCI format.
            board_state: Current board state.
            
        Returns:
            True if move gives check.
        """
        try:
            chess_move = chess.Move.from_uci(move)
            return board_state.board.gives_check(chess_move)
        except (ValueError, chess.InvalidMoveError):
            return False
    
    def find_matching_move(self, from_sq: str, to_sq: str,
                          board_state: BoardState) -> Optional[str]:
        """Find a legal move matching the given squares.
        
        Args:
            from_sq: Source square.
            to_sq: Destination square.
            board_state: Current board state.
            
        Returns:
            Legal move in UCI format, or None if no match.
        """
        try:
            from_chess = chess.parse_square(from_sq)
            to_chess = chess.parse_square(to_sq)
        except ValueError:
            return None
        
        for move in board_state.board.legal_moves:
            if move.from_square == from_chess and move.to_square == to_chess:
                return move.uci()
        
        return None
