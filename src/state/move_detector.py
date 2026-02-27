"""Move detection from board state changes."""
import chess
import numpy as np
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

from src.utils.logging_setup import get_logger
from src.utils.helpers import index_to_square

logger = get_logger(__name__)


@dataclass
class DetectedMove:
    """Represents a detected move."""
    uci: str
    from_square: str
    to_square: str
    is_capture: bool = False
    is_castling: bool = False
    is_en_passant: bool = False
    is_promotion: bool = False
    promotion_piece: Optional[str] = None
    confidence: float = 1.0


class MoveDetector:
    """Detects chess moves by comparing board states."""
    
    def __init__(self):
        """Initialize the move detector."""
        self._previous_grid: Optional[np.ndarray] = None
    
    def detect_move(self, prev_grid: np.ndarray, curr_grid: np.ndarray,
                   board: chess.Board) -> Optional[DetectedMove]:
        """Detect the move made between two board states.
        
        Args:
            prev_grid: Previous 8x8 occupancy grid (1=white, -1=black, 0=empty).
            curr_grid: Current 8x8 occupancy grid.
            board: Current chess.Board (before the detected move).
            
        Returns:
            DetectedMove if a valid move detected, None otherwise.
        """
        # Find squares that changed
        changes = self._find_changes(prev_grid, curr_grid)
        
        if not changes:
            return None
        
        logger.debug(f"Detected changes: {changes}")
        
        # Determine whose turn
        is_white_turn = board.turn == chess.WHITE
        own_color = 1 if is_white_turn else -1
        
        # Analyze changes to determine the move
        emptied = changes['emptied']  # Squares that became empty
        filled = changes['filled']    # Squares that became occupied
        
        # Check for various move types
        
        # Standard move: one emptied, one filled
        if len(emptied) == 1 and len(filled) == 1:
            from_sq = emptied[0]
            to_sq = filled[0]
            
            # Verify this could be the moving player's piece
            from_row, from_col = self._square_to_rc(from_sq)
            to_row, to_col = self._square_to_rc(to_sq)
            
            if prev_grid[from_row, from_col] == own_color:
                # Check for pawn promotion
                is_promotion = self._is_promotion_move(from_sq, to_sq, board)
                promotion = 'q' if is_promotion else None  # Default to queen
                
                return DetectedMove(
                    uci=f"{from_sq}{to_sq}{promotion or ''}",
                    from_square=from_sq,
                    to_square=to_sq,
                    is_promotion=is_promotion,
                    promotion_piece=promotion
                )
        
        # Capture: one emptied (piece moved from), one changed color
        if len(emptied) == 1 and len(changes['changed_color']) == 1:
            from_sq = emptied[0]
            to_sq = changes['changed_color'][0]
            
            from_row, from_col = self._square_to_rc(from_sq)
            if prev_grid[from_row, from_col] == own_color:
                is_promotion = self._is_promotion_move(from_sq, to_sq, board)
                promotion = 'q' if is_promotion else None
                
                return DetectedMove(
                    uci=f"{from_sq}{to_sq}{promotion or ''}",
                    from_square=from_sq,
                    to_square=to_sq,
                    is_capture=True,
                    is_promotion=is_promotion,
                    promotion_piece=promotion
                )
        
        # Castling: two squares emptied, two filled (king and rook)
        if len(emptied) == 2 and len(filled) == 2:
            castle_move = self._detect_castling(emptied, filled, board)
            if castle_move:
                return castle_move
        
        # En passant: two emptied (pawn + captured), one filled
        if len(emptied) == 2 and len(filled) == 1:
            ep_move = self._detect_en_passant(emptied, filled, board)
            if ep_move:
                return ep_move
        
        # If we can't determine the exact move, try matching against legal moves
        return self._match_against_legal_moves(changes, board)
    
    def _find_changes(self, prev: np.ndarray, curr: np.ndarray) -> dict:
        """Find all changes between two board states."""
        changes = {
            'emptied': [],       # Was occupied, now empty
            'filled': [],        # Was empty, now occupied  
            'changed_color': [], # Changed from one color to another
            'all_changed': []    # All squares that changed
        }
        
        for row in range(8):
            for col in range(8):
                prev_val = prev[row, col]
                curr_val = curr[row, col]
                
                if prev_val != curr_val:
                    square = index_to_square(row, col)
                    changes['all_changed'].append(square)
                    
                    if prev_val != 0 and curr_val == 0:
                        changes['emptied'].append(square)
                    elif prev_val == 0 and curr_val != 0:
                        changes['filled'].append(square)
                    elif prev_val != 0 and curr_val != 0:
                        changes['changed_color'].append(square)
        
        return changes
    
    def _square_to_rc(self, square: str) -> Tuple[int, int]:
        """Convert square name to row, col indices."""
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        return row, col
    
    def _is_promotion_move(self, from_sq: str, to_sq: str, board: chess.Board) -> bool:
        """Check if a move would be a pawn promotion."""
        from_chess_sq = chess.parse_square(from_sq)
        piece = board.piece_at(from_chess_sq)
        
        if piece and piece.piece_type == chess.PAWN:
            to_rank = int(to_sq[1])
            if piece.color == chess.WHITE and to_rank == 8:
                return True
            if piece.color == chess.BLACK and to_rank == 1:
                return True
        
        return False
    
    def _detect_castling(self, emptied: List[str], filled: List[str],
                        board: chess.Board) -> Optional[DetectedMove]:
        """Detect if the changes represent castling."""
        # King-side castling: e1->g1 (white) or e8->g8 (black)
        # Queen-side castling: e1->c1 (white) or e8->c8 (black)
        
        white_kingside = {'e1', 'h1'}, {'f1', 'g1'}
        white_queenside = {'e1', 'a1'}, {'c1', 'd1'}
        black_kingside = {'e8', 'h8'}, {'f8', 'g8'}
        black_queenside = {'e8', 'a8'}, {'c8', 'd8'}
        
        emptied_set = set(emptied)
        filled_set = set(filled)
        
        if emptied_set == white_kingside[0] and filled_set == white_kingside[1]:
            return DetectedMove(uci='e1g1', from_square='e1', to_square='g1', is_castling=True)
        if emptied_set == white_queenside[0] and filled_set == white_queenside[1]:
            return DetectedMove(uci='e1c1', from_square='e1', to_square='c1', is_castling=True)
        if emptied_set == black_kingside[0] and filled_set == black_kingside[1]:
            return DetectedMove(uci='e8g8', from_square='e8', to_square='g8', is_castling=True)
        if emptied_set == black_queenside[0] and filled_set == black_queenside[1]:
            return DetectedMove(uci='e8c8', from_square='e8', to_square='c8', is_castling=True)
        
        return None
    
    def _detect_en_passant(self, emptied: List[str], filled: List[str],
                          board: chess.Board) -> Optional[DetectedMove]:
        """Detect if the changes represent en passant."""
        if len(emptied) != 2 or len(filled) != 1:
            return None
        
        to_sq = filled[0]
        to_file = to_sq[0]
        to_rank = int(to_sq[1])
        
        # En passant target must be on rank 3 (black) or 6 (white)
        if to_rank not in (3, 6):
            return None
        
        # Find the pawn that moved
        for from_sq in emptied:
            from_file = from_sq[0]
            from_rank = int(from_sq[1])
            
            # Check if this is the moving pawn (diagonal move)
            if abs(ord(from_file) - ord(to_file)) == 1:
                if (from_rank == 4 and to_rank == 3) or (from_rank == 5 and to_rank == 6):
                    # Verify it matches the board's en passant square
                    if board.ep_square and chess.square_name(board.ep_square) == to_sq:
                        return DetectedMove(
                            uci=f"{from_sq}{to_sq}",
                            from_square=from_sq,
                            to_square=to_sq,
                            is_capture=True,
                            is_en_passant=True
                        )
        
        return None
    
    def _match_against_legal_moves(self, changes: dict, 
                                   board: chess.Board) -> Optional[DetectedMove]:
        """Try to match changes against legal moves."""
        changed_squares = set(changes['all_changed'])
        
        for move in board.legal_moves:
            move_squares = {chess.square_name(move.from_square), 
                          chess.square_name(move.to_square)}
            
            # For castling, include rook squares
            if board.is_castling(move):
                if board.is_kingside_castling(move):
                    if move.from_square == chess.E1:
                        move_squares.update({'h1', 'f1', 'g1'})
                    else:
                        move_squares.update({'h8', 'f8', 'g8'})
                else:
                    if move.from_square == chess.E1:
                        move_squares.update({'a1', 'd1', 'c1'})
                    else:
                        move_squares.update({'a8', 'd8', 'c8'})
            
            # For en passant, include captured pawn square
            if board.is_en_passant(move):
                captured_sq = chess.square_name(move.to_square)
                # The captured pawn is on the same file as to_square but different rank
                if board.turn == chess.WHITE:
                    move_squares.add(captured_sq[0] + '5')
                else:
                    move_squares.add(captured_sq[0] + '4')
            
            # Check if this move matches the detected changes
            if changed_squares.issubset(move_squares) or move_squares.issubset(changed_squares):
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                
                return DetectedMove(
                    uci=move.uci(),
                    from_square=from_sq,
                    to_square=to_sq,
                    is_capture=board.is_capture(move),
                    is_castling=board.is_castling(move),
                    is_en_passant=board.is_en_passant(move),
                    is_promotion=move.promotion is not None,
                    promotion_piece=chess.piece_symbol(move.promotion) if move.promotion else None,
                    confidence=0.8  # Lower confidence for matched moves
                )
        
        logger.warning(f"Could not match changes to legal move: {changes}")
        return None
    
    def set_previous_state(self, grid: np.ndarray) -> None:
        """Set the previous state for comparison."""
        self._previous_grid = grid.copy()
    
    def get_changed_squares(self, prev: np.ndarray, curr: np.ndarray) -> List[str]:
        """Get list of squares that changed between two states."""
        changes = self._find_changes(prev, curr)
        return changes['all_changed']
