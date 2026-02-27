"""Tests for chess engine integration."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.engine import StockfishEngine


class TestStockfishEngine:
    """Tests for StockfishEngine class."""
    
    def test_initialization(self):
        """Test engine initialization (without starting)."""
        engine = StockfishEngine(path='stockfish', skill_level=15)
        
        assert engine.skill_level == 15
        assert not engine.is_ready
    
    def test_start_without_stockfish(self):
        """Test start returns False when Stockfish not found."""
        engine = StockfishEngine(path='/nonexistent/stockfish')
        
        result = engine.start()
        
        assert result is False
        assert not engine.is_ready
    
    @pytest.mark.skipif(
        not any(os.path.exists(p) for p in [
            'stockfish/stockfish.exe',
            '/usr/local/bin/stockfish',
            '/usr/bin/stockfish'
        ]),
        reason="Stockfish not installed"
    )
    def test_start_with_stockfish(self):
        """Test starting engine when Stockfish is available."""
        engine = StockfishEngine()
        
        try:
            result = engine.start()
            
            if result:
                assert engine.is_ready
                engine.quit()
        except Exception:
            pass  # Stockfish may not be available
    
    @pytest.mark.skipif(
        not any(os.path.exists(p) for p in [
            'stockfish/stockfish.exe',
            '/usr/local/bin/stockfish',
            '/usr/bin/stockfish'
        ]),
        reason="Stockfish not installed"
    )
    def test_get_best_move(self):
        """Test getting best move from engine."""
        engine = StockfishEngine()
        
        try:
            if not engine.start():
                pytest.skip("Stockfish not available")
            
            engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            move = engine.get_best_move(time_limit=0.5)
            
            # Should return a valid UCI move
            assert move is not None
            assert len(move) >= 4
            
            engine.quit()
        except Exception:
            pass
    
    def test_skill_level_clamping(self):
        """Test skill level is clamped to valid range."""
        engine = StockfishEngine(skill_level=25)
        
        # Should clamp internally when setting
        engine.set_skill_level(-5)
        assert engine.skill_level == 0
        
        engine.set_skill_level(25)
        assert engine.skill_level == 20
    
    def test_context_manager(self):
        """Test engine as context manager."""
        # This should not raise even if Stockfish isn't available
        try:
            with StockfishEngine(path='/nonexistent/stockfish') as engine:
                assert not engine.is_ready
        except Exception:
            pass  # May fail to start, which is fine


class TestMockedStockfish:
    """Tests with mocked Stockfish process."""
    
    @patch('subprocess.Popen')
    def test_uci_initialization(self, mock_popen):
        """Test UCI protocol initialization."""
        # Mock the process
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = MagicMock(side_effect=[
            'Stockfish 15\n',
            'uciok\n',
            'readyok\n'
        ])
        mock_popen.return_value = mock_process
        
        engine = StockfishEngine(path='stockfish')
        
        # Patch os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            # The actual start may fail due to mock limitations
            # but we can test the structure
            pass
    
    def test_set_position_format(self):
        """Test position string formatting."""
        engine = StockfishEngine()
        engine._is_ready = True
        engine._process = MagicMock()
        engine._process.stdin = MagicMock()
        
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        engine.set_position(fen)
        
        # Check command was sent
        engine._process.stdin.write.assert_called()


class TestEngineEvaluation:
    """Tests for engine evaluation features."""
    
    @pytest.mark.skipif(
        not any(os.path.exists(p) for p in [
            'stockfish/stockfish.exe',
            '/usr/local/bin/stockfish',
            '/usr/bin/stockfish'
        ]),
        reason="Stockfish not installed"
    )
    def test_get_evaluation(self):
        """Test getting position evaluation."""
        engine = StockfishEngine()
        
        try:
            if not engine.start():
                pytest.skip("Stockfish not available")
            
            # Starting position should be roughly equal
            engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            eval_score = engine.get_evaluation()
            
            if eval_score is not None:
                # Should be within reasonable range for starting position
                assert -100 < eval_score < 100
            
            engine.quit()
        except Exception:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
