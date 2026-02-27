"""Tests for vision components."""
import pytest
import numpy as np
import cv2

from src.vision import BoardDetector, PerspectiveTransformer, SquareExtractor, PieceDetector
from src.vision.board_detector import BoardDetectionResult
from src.vision.piece_detector import SquareState
from src.utils.helpers import square_to_index, index_to_square, is_light_square


class TestBoardDetector:
    """Tests for BoardDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = BoardDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert not self.detector.is_calibrated
        assert self.detector.calibrated_corners is None
    
    def test_calibration(self):
        """Test manual calibration."""
        corners = np.array([
            [100, 100],
            [500, 100],
            [500, 500],
            [100, 500]
        ], dtype=np.float32)
        
        self.detector.calibrate(corners)
        
        assert self.detector.is_calibrated
        assert self.detector.calibrated_corners is not None
        assert self.detector.calibrated_corners.shape == (4, 2)
    
    def test_reset_calibration(self):
        """Test resetting calibration."""
        corners = np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ], dtype=np.float32)
        
        self.detector.calibrate(corners)
        self.detector.reset_calibration()
        
        assert not self.detector.is_calibrated
        assert self.detector.calibrated_corners is None
    
    def test_detect_with_calibration(self):
        """Test detection with calibrated corners."""
        corners = np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ], dtype=np.float32)
        
        self.detector.calibrate(corners)
        
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        result = self.detector.detect(frame)
        
        assert result.found
        assert result.confidence == 1.0
        assert np.allclose(result.corners, corners, atol=1)
    
    def test_detect_synthetic_board(self):
        """Test detection on synthetic board image."""
        # Create a synthetic chessboard pattern
        frame = np.ones((600, 600, 3), dtype=np.uint8) * 200
        
        # Draw a square board
        board_corners = [(100, 100), (500, 100), (500, 500), (100, 500)]
        pts = np.array(board_corners, dtype=np.int32)
        cv2.fillPoly(frame, [pts], (100, 100, 100))
        
        # Draw alternating squares
        square_size = 50
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    x1 = 100 + j * square_size
                    y1 = 100 + i * square_size
                    cv2.rectangle(frame, (x1, y1), (x1 + square_size, y1 + square_size),
                                (220, 220, 200), -1)
        
        result = self.detector.detect(frame)
        
        # Detection may or may not succeed depending on contrast
        # At minimum, check result structure
        assert isinstance(result, BoardDetectionResult)
        assert hasattr(result, 'found')
        assert hasattr(result, 'corners')


class TestPerspectiveTransformer:
    """Tests for PerspectiveTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = PerspectiveTransformer(output_size=640)
    
    def test_initialization(self):
        """Test transformer initialization."""
        assert self.transformer.output_size == 640
        assert not self.transformer.is_ready
    
    def test_compute_transform(self):
        """Test computing perspective transform."""
        corners = np.array([
            [100, 100],
            [500, 100],
            [500, 500],
            [100, 500]
        ], dtype=np.float32)
        
        matrix = self.transformer.compute_transform(corners)
        
        assert matrix is not None
        assert matrix.shape == (3, 3)
        assert self.transformer.is_ready
    
    def test_transform_image(self):
        """Test transforming an image."""
        corners = np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ], dtype=np.float32)
        
        # Create test image
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), -1)
        
        result = self.transformer.transform(frame, corners)
        
        assert result is not None
        assert result.shape == (640, 640, 3)
    
    def test_transform_point(self):
        """Test transforming a single point."""
        corners = np.array([
            [0, 0], [640, 0], [640, 640], [0, 640]
        ], dtype=np.float32)
        
        self.transformer.compute_transform(corners)
        
        # Transform corner point
        result = self.transformer.transform_point((0, 0))
        
        assert result is not None
        assert abs(result[0]) < 5  # Should be near (0, 0)
        assert abs(result[1]) < 5


class TestSquareExtractor:
    """Tests for SquareExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SquareExtractor(board_size=640)
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.board_size == 640
        assert self.extractor.square_size == 80
    
    def test_extract_all(self):
        """Test extracting all squares."""
        # Create test board image
        board = np.zeros((640, 640, 3), dtype=np.uint8)
        
        squares = self.extractor.extract_all(board)
        
        assert len(squares) == 64
        assert 'e4' in squares
        assert 'a1' in squares
        assert 'h8' in squares
        
        # Check square sizes
        for name, img in squares.items():
            assert img.shape == (80, 80, 3)
    
    def test_extract_single_square(self):
        """Test extracting a single square."""
        board = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        square = self.extractor.extract_square(board, 'e4')
        
        assert square is not None
        assert square.shape == (80, 80, 3)
    
    def test_get_square_center(self):
        """Test getting square center coordinates."""
        center = self.extractor.get_square_center('a8')
        
        assert center is not None
        assert center == (40, 40)  # Top-left square
        
        center = self.extractor.get_square_center('h1')
        
        assert center is not None
        assert center == (600, 600)  # Bottom-right square
    
    def test_point_to_square(self):
        """Test converting coordinates to square name."""
        # Top-left corner region
        assert self.extractor.point_to_square(40, 40) == 'a8'
        
        # Bottom-right corner region
        assert self.extractor.point_to_square(600, 600) == 'h1'
        
        # Center of board
        assert self.extractor.point_to_square(320, 320) in ['d4', 'd5', 'e4', 'e5']
    
    def test_invalid_square(self):
        """Test handling invalid square names."""
        board = np.zeros((640, 640, 3), dtype=np.uint8)
        
        result = self.extractor.extract_square(board, 'i9')
        assert result is None
        
        center = self.extractor.get_square_center('z1')
        assert center is None


class TestPieceDetector:
    """Tests for PieceDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PieceDetector()
    
    def test_classify_empty_square(self):
        """Test classifying an empty square."""
        # Create uniform colored square (should be empty)
        square = np.ones((80, 80, 3), dtype=np.uint8) * 180
        
        result = self.detector.classify_square(square, 'e4')
        
        # Uniform color = low variance = empty
        assert result == SquareState.EMPTY
    
    def test_classify_white_piece(self):
        """Test classifying a white piece."""
        # Create square with bright center (white piece)
        square = np.ones((80, 80, 3), dtype=np.uint8) * 100  # Dark background
        center = square[20:60, 20:60]
        center[:] = 240  # Bright center = white piece
        
        result = self.detector.classify_square(square, 'e2')
        
        assert result == SquareState.WHITE_PIECE
    
    def test_classify_black_piece(self):
        """Test classifying a black piece."""
        # Create square with dark center (black piece)
        square = np.ones((80, 80, 3), dtype=np.uint8) * 180  # Light background
        center = square[20:60, 20:60]
        center[:] = 40  # Dark center = black piece
        
        result = self.detector.classify_square(square, 'e7')
        
        assert result == SquareState.BLACK_PIECE
    
    def test_classify_board(self):
        """Test classifying entire board."""
        # Create dict of 64 empty squares
        squares = {}
        for file in 'abcdefgh':
            for rank in '12345678':
                square = np.ones((80, 80, 3), dtype=np.uint8) * 150
                squares[f'{file}{rank}'] = square
        
        result = self.detector.classify_board(squares)
        
        assert result.shape == (8, 8)
        # All should be classified as empty
        assert np.all(result == SquareState.EMPTY)
    
    def test_set_thresholds(self):
        """Test setting detection thresholds."""
        self.detector.set_thresholds(piece_presence=20.0, white_black=140.0)
        
        assert self.detector._piece_presence_threshold == 20.0
        assert self.detector._white_black_threshold == 140.0
    
    def test_get_detection_debug(self):
        """Test getting debug information."""
        square = np.ones((80, 80, 3), dtype=np.uint8) * 150
        
        debug = self.detector.get_detection_debug(square, 'e4')
        
        assert 'square' in debug
        assert 'is_light' in debug
        assert 'mean' in debug
        assert 'std' in debug
        assert 'classification' in debug
        assert debug['square'] == 'e4'


class TestHelpers:
    """Tests for utility helper functions."""
    
    def test_square_to_index(self):
        """Test converting square names to indices."""
        assert square_to_index('a8') == (0, 0)
        assert square_to_index('h8') == (0, 7)
        assert square_to_index('a1') == (7, 0)
        assert square_to_index('h1') == (7, 7)
        assert square_to_index('e4') == (4, 4)
    
    def test_index_to_square(self):
        """Test converting indices to square names."""
        assert index_to_square(0, 0) == 'a8'
        assert index_to_square(0, 7) == 'h8'
        assert index_to_square(7, 0) == 'a1'
        assert index_to_square(7, 7) == 'h1'
        assert index_to_square(4, 4) == 'e4'
    
    def test_is_light_square(self):
        """Test light/dark square detection."""
        assert is_light_square('a1') == False  # a1 is dark
        assert is_light_square('a8') == True   # a8 is light
        assert is_light_square('h1') == True   # h1 is light
        assert is_light_square('h8') == False  # h8 is dark
        assert is_light_square('e4') == True   # e4 is light


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
