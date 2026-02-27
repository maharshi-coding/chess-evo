"""Tests for board state and move detection."""
import pytest
import numpy as np
import chess

from src.state import BoardState, MoveDetector, MoveValidator
from src.state.move_validator import MoveType


class TestBoardState:
    """Tests for BoardState class."""
    
    def test_initial_position(self):
        """Test initial board state."""
        state = BoardState()
        assert state.fen == chess.STARTING_FEN
        assert state.turn == chess.WHITE
        assert state.fullmove_number == 1
    
    def test_from_fen(self):
        """Test creating board from FEN."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        state = BoardState.from_fen(fen)
        assert state.turn == chess.BLACK
        assert state.get_piece_at('e4') == 'P'
        assert state.get_piece_at('e2') is None
    
    def test_get_piece_at(self):
        """Test getting pieces at squares."""
        state = BoardState()
        assert state.get_piece_at('e1') == 'K'  # White king
        assert state.get_piece_at('e8') == 'k'  # Black king
        assert state.get_piece_at('e2') == 'P'  # White pawn
        assert state.get_piece_at('e4') is None  # Empty
    
    def test_make_move(self):
        """Test making moves."""
        state = BoardState()
        
        # Legal move
        result = state.make_move('e2e4')
        assert result is True
        assert state.get_piece_at('e4') == 'P'
        assert state.get_piece_at('e2') is None
        assert state.turn == chess.BLACK
        
        # Black responds
        result = state.make_move('e7e5')
        assert result is True
        assert state.turn == chess.WHITE
    
    def test_illegal_move(self):
        """Test that illegal moves are rejected."""
        state = BoardState()
        
        # Try to move to occupied square
        result = state.make_move('e2e1')
        assert result is False
        
        # Try to move opponent's piece
        result = state.make_move('e7e5')
        assert result is False
        assert state.turn == chess.WHITE
    
    def test_unmake_move(self):
        """Test undoing moves."""
        state = BoardState()
        state.make_move('e2e4')
        
        undone = state.unmake_move()
        assert undone == 'e2e4'
        assert state.get_piece_at('e2') == 'P'
        assert state.get_piece_at('e4') is None
        assert state.turn == chess.WHITE
    
    def test_get_legal_moves(self):
        """Test getting legal moves."""
        state = BoardState()
        moves = state.get_legal_moves()
        
        # Initial position has 20 legal moves
        assert len(moves) == 20
        assert 'e2e4' in moves
        assert 'g1f3' in moves
    
    def test_to_occupancy_grid(self):
        """Test conversion to occupancy grid."""
        state = BoardState()
        grid = state.to_occupancy_grid()
        
        assert grid.shape == (8, 8)
        # White back rank (row 7)
        assert all(grid[7, :] == 1)
        # Black back rank (row 0)
        assert all(grid[0, :] == -1)
        # Empty middle
        assert all(grid[3, :] == 0)
        assert all(grid[4, :] == 0)
    
    def test_copy(self):
        """Test copying board state."""
        state = BoardState()
        state.make_move('e2e4')
        
        copy = state.copy()
        assert copy.fen == state.fen
        
        # Modifying copy shouldn't affect original
        copy.make_move('e7e5')
        assert state.fen != copy.fen


class TestMoveDetector:
    """Tests for MoveDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MoveDetector()
        self.board = chess.Board()
    
    def test_detect_simple_move(self):
        """Test detecting a simple pawn move."""
        # Starting position occupancy
        prev = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1],  # Row 0: black pieces
            [-1, -1, -1, -1, -1, -1, -1, -1],  # Row 1: black pawns
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],   # Row 6: white pawns
            [1, 1, 1, 1, 1, 1, 1, 1],   # Row 7: white pieces
        ], dtype=np.int8)
        
        # After e2-e4 (col 4, row 6 -> row 4)
        curr = prev.copy()
        curr[6, 4] = 0   # e2 now empty
        curr[4, 4] = 1   # e4 now occupied
        
        detected = self.detector.detect_move(prev, curr, self.board)
        
        assert detected is not None
        assert detected.from_square == 'e2'
        assert detected.to_square == 'e4'
        assert detected.uci == 'e2e4'
    
    def test_detect_capture(self):
        """Test detecting a capture."""
        # Set up position where capture is possible
        self.board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        
        prev = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, -1],  # d7 pawn missing
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],  # d5 has black pawn
            [0, 0, 0, 0, 1, 0, 0, 0],   # e4 has white pawn
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int8)
        
        # After exd5
        curr = prev.copy()
        curr[4, 4] = 0   # e4 now empty
        curr[3, 3] = 1   # d5 now white (capture)
        
        detected = self.detector.detect_move(prev, curr, self.board)
        
        assert detected is not None
        assert detected.uci == 'e4d5'
    
    def test_detect_castling(self):
        """Test detecting castling."""
        # Position where castling is legal
        self.board = chess.Board("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        
        prev = np.array([
            [-1, 0, 0, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],  # Rooks on a1, h1, king on e1
        ], dtype=np.int8)
        
        # After kingside castling
        curr = prev.copy()
        curr[7, 4] = 0  # e1 empty (king moved)
        curr[7, 7] = 0  # h1 empty (rook moved)
        curr[7, 6] = 1  # g1 has king
        curr[7, 5] = 1  # f1 has rook
        
        detected = self.detector.detect_move(prev, curr, self.board)
        
        assert detected is not None
        assert detected.is_castling
        assert detected.uci == 'e1g1'
    
    def test_get_changed_squares(self):
        """Test getting changed squares."""
        prev = np.zeros((8, 8), dtype=np.int8)
        curr = np.zeros((8, 8), dtype=np.int8)
        
        prev[6, 4] = 1   # e2 occupied
        curr[4, 4] = 1   # e4 occupied
        
        changed = self.detector.get_changed_squares(prev, curr)
        
        assert 'e2' in changed
        assert 'e4' in changed
        assert len(changed) == 2


class TestMoveValidator:
    """Tests for MoveValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MoveValidator()
    
    def test_validate_legal_move(self):
        """Test validating a legal move."""
        state = BoardState()
        result = self.validator.validate('e2e4', state)
        
        assert result.is_valid
        assert result.move == 'e2e4'
        assert result.move_type == MoveType.NORMAL
    
    def test_validate_illegal_move(self):
        """Test validating an illegal move."""
        state = BoardState()
        result = self.validator.validate('e2e5', state)  # Pawn can't move 3 squares
        
        assert not result.is_valid
        assert result.error is not None
    
    def test_validate_capture(self):
        """Test validating a capture."""
        state = BoardState.from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        result = self.validator.validate('e4d5', state)
        
        assert result.is_valid
        assert result.move_type == MoveType.CAPTURE
    
    def test_validate_castling(self):
        """Test validating castling."""
        state = BoardState.from_fen("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        result = self.validator.validate('e1g1', state)
        
        assert result.is_valid
        assert result.move_type == MoveType.CASTLE_KINGSIDE
    
    def test_validate_promotion(self):
        """Test validating pawn promotion."""
        state = BoardState.from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        result = self.validator.validate('a7a8q', state)
        
        assert result.is_valid
        assert result.move_type == MoveType.PROMOTION
    
    def test_auto_promotion(self):
        """Test auto-promotion when not specified."""
        state = BoardState.from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        result = self.validator.validate('a7a8', state)
        
        # Should auto-add queen promotion
        assert result.is_valid
        assert result.move == 'a7a8q'
    
    def test_get_legal_moves(self):
        """Test getting all legal moves."""
        state = BoardState()
        moves = self.validator.get_legal_moves(state)
        
        assert len(moves) == 20
    
    def test_get_legal_moves_from_square(self):
        """Test getting legal moves from a specific square."""
        state = BoardState()
        moves = self.validator.get_legal_moves_from(state, 'e2')
        
        assert 'e2e3' in moves
        assert 'e2e4' in moves
        assert len(moves) == 2
    
    def test_gives_check(self):
        """Test checking if move gives check."""
        state = BoardState.from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        
        # Qh5 doesn't give check
        assert not self.validator.gives_check('d1h5', state)
        
        # But if we play Qh5, then Qxf7 would be checkmate
        state.make_move('d1h5')
        assert self.validator.gives_check('h5f7', state)


class TestCheckAndMate:
    """Tests for check, checkmate, and stalemate detection."""
    
    def test_is_check(self):
        """Test check detection."""
        # Position with black in check after Qh5-f7
        state = BoardState.from_fen("rnbqkbnr/pppp1Qpp/8/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 2")
        assert state.is_check()
    
    def test_is_checkmate(self):
        """Test checkmate detection (Scholar's mate)."""
        state = BoardState.from_fen("rnbqkb1r/pppp1Qpp/5n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        # This isn't quite checkmate, let's use a real checkmate position
        state = BoardState.from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert state.is_checkmate()
    
    def test_is_stalemate(self):
        """Test stalemate detection."""
        # Classic stalemate position
        state = BoardState.from_fen("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        # This is actually not stalemate, king can move. Let's use proper stalemate:
        state = BoardState.from_fen("k7/8/K7/8/8/8/8/1Q6 b - - 0 1")
        # Still not stalemate. Proper example:
        state = BoardState.from_fen("7k/8/6K1/8/8/8/8/7Q b - - 0 1")
        # King trapped in corner with queen covering all escape squares
        # Actually this still allows Kg8. True stalemate example:
        state = BoardState.from_fen("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
        assert state.is_stalemate()
    
    def test_game_over(self):
        """Test game over detection."""
        # Normal position - not game over
        state = BoardState()
        assert not state.is_game_over()
        
        # Checkmate position - game over
        state = BoardState.from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert state.is_game_over()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
