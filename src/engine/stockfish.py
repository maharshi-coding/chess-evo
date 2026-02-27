"""Stockfish chess engine interface."""
import subprocess
import os
from typing import Optional, Tuple
from pathlib import Path
import threading
import queue
import time

from src.config import config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class StockfishEngine:
    """Interface to the Stockfish chess engine using UCI protocol."""
    
    def __init__(self, path: Optional[str] = None, skill_level: int = 10):
        """Initialize the Stockfish engine.
        
        Args:
            path: Path to Stockfish executable. If None, uses config.
            skill_level: Engine skill level (0-20).
        """
        self.path = path or config.engine.get('path', 'stockfish')
        self.skill_level = skill_level
        self.move_time = config.engine.get('move_time', 1.0)
        self.depth = config.engine.get('depth', 15)
        
        self._process: Optional[subprocess.Popen] = None
        self._is_ready = False
        self._current_fen: Optional[str] = None
        
    def start(self) -> bool:
        """Start the Stockfish engine.
        
        Returns:
            True if engine started successfully.
        """
        try:
            # Check if path exists
            if not os.path.exists(self.path):
                # Try common locations
                common_paths = [
                    'stockfish/stockfish.exe',
                    'stockfish/stockfish',
                    '/usr/local/bin/stockfish',
                    '/usr/bin/stockfish',
                    'C:/stockfish/stockfish.exe'
                ]
                
                for p in common_paths:
                    if os.path.exists(p):
                        self.path = p
                        break
                else:
                    logger.warning(f"Stockfish not found at {self.path}")
                    logger.info("Download from: https://stockfishchess.org/download/")
                    return False
            
            self._process = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Initialize UCI
            self._send_command('uci')
            response = self._read_until('uciok', timeout=5.0)
            
            if 'uciok' not in response:
                logger.error("Failed to initialize UCI")
                self.quit()
                return False
            
            # Set options
            self._send_command(f'setoption name Skill Level value {self.skill_level}')
            
            # Ready check
            self._send_command('isready')
            response = self._read_until('readyok', timeout=5.0)
            
            if 'readyok' not in response:
                logger.error("Engine not ready")
                self.quit()
                return False
            
            self._is_ready = True
            logger.info(f"Stockfish started: {self.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Stockfish: {e}")
            return False
    
    def quit(self) -> None:
        """Stop the Stockfish engine."""
        if self._process:
            try:
                self._send_command('quit')
                self._process.wait(timeout=2.0)
            except:
                self._process.kill()
            finally:
                self._process = None
                self._is_ready = False
        
        logger.info("Stockfish stopped")
    
    def set_position(self, fen: str) -> None:
        """Set the current position.
        
        Args:
            fen: Position in FEN format.
        """
        if not self._is_ready:
            logger.warning("Engine not ready")
            return
        
        self._send_command(f'position fen {fen}')
        self._current_fen = fen
    
    def set_position_with_moves(self, fen: str, moves: list) -> None:
        """Set position with move history.
        
        Args:
            fen: Starting position in FEN.
            moves: List of moves in UCI format.
        """
        if not self._is_ready:
            return
        
        moves_str = ' '.join(moves)
        self._send_command(f'position fen {fen} moves {moves_str}')
        self._current_fen = fen
    
    def get_best_move(self, time_limit: Optional[float] = None,
                     depth: Optional[int] = None) -> Optional[str]:
        """Get the best move for the current position.
        
        Args:
            time_limit: Time limit in seconds. If None, uses config.
            depth: Search depth. If None, uses config.
            
        Returns:
            Best move in UCI format, or None if failed.
        """
        if not self._is_ready:
            logger.warning("Engine not ready")
            return None
        
        time_limit = time_limit or self.move_time
        depth = depth or self.depth
        
        # Send go command
        movetime_ms = int(time_limit * 1000)
        self._send_command(f'go movetime {movetime_ms} depth {depth}')
        
        # Read until bestmove
        response = self._read_until('bestmove', timeout=time_limit + 5.0)
        
        # Parse bestmove
        for line in response.split('\n'):
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    move = parts[1]
                    logger.info(f"Engine move: {move}")
                    return move
        
        logger.warning("No bestmove found in response")
        return None
    
    def get_evaluation(self) -> Optional[float]:
        """Get position evaluation in centipawns.
        
        Returns:
            Evaluation from White's perspective, or None if unavailable.
        """
        if not self._is_ready:
            return None
        
        # Quick evaluation search
        self._send_command('go depth 10')
        response = self._read_until('bestmove', timeout=3.0)
        
        # Parse score from info lines
        for line in response.split('\n'):
            if 'score cp' in line:
                parts = line.split()
                try:
                    cp_idx = parts.index('cp')
                    return int(parts[cp_idx + 1])
                except (ValueError, IndexError):
                    pass
            elif 'score mate' in line:
                parts = line.split()
                try:
                    mate_idx = parts.index('mate')
                    mate_in = int(parts[mate_idx + 1])
                    # Return large value for mate
                    return 10000 if mate_in > 0 else -10000
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def set_skill_level(self, level: int) -> None:
        """Set engine skill level.
        
        Args:
            level: Skill level from 0 (weakest) to 20 (strongest).
        """
        level = max(0, min(20, level))
        self.skill_level = level
        
        if self._is_ready:
            self._send_command(f'setoption name Skill Level value {level}')
            logger.info(f"Skill level set to {level}")
    
    def _send_command(self, command: str) -> None:
        """Send a command to the engine."""
        if self._process and self._process.stdin:
            self._process.stdin.write(command + '\n')
            self._process.stdin.flush()
            logger.debug(f"Sent: {command}")
    
    def _read_until(self, target: str, timeout: float = 5.0) -> str:
        """Read engine output until target string found.
        
        Args:
            target: String to look for.
            timeout: Maximum time to wait.
            
        Returns:
            All output read.
        """
        if not self._process or not self._process.stdout:
            return ''
        
        output = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Non-blocking read
                import select
                if hasattr(select, 'select'):
                    readable, _, _ = select.select([self._process.stdout], [], [], 0.1)
                    if not readable:
                        continue
                
                line = self._process.stdout.readline()
                if line:
                    output.append(line.strip())
                    logger.debug(f"Recv: {line.strip()}")
                    
                    if target in line:
                        break
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.debug(f"Read error: {e}")
                time.sleep(0.01)
        
        return '\n'.join(output)
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready."""
        return self._is_ready
    
    def __enter__(self) -> 'StockfishEngine':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()
