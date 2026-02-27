"""Game controller - main orchestration of the chess game."""
import numpy as np
import chess
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

from src.config import config
from src.utils.logging_setup import get_logger
from src.camera import FrameCapture
from src.vision import BoardDetector, PerspectiveTransformer, SquareExtractor, PieceDetector
from src.state import BoardState, MoveDetector, MoveValidator
from src.engine import StockfishEngine
from .states import GameState, PlayerColor, is_game_over_state

logger = get_logger(__name__)


@dataclass
class GameEvent:
    """Represents a game event for history tracking."""
    timestamp: datetime
    event_type: str
    data: dict = field(default_factory=dict)


class GameController:
    """Main game controller that orchestrates all components."""
    
    def __init__(self, player_color: str = 'white'):
        """Initialize the game controller.
        
        Args:
            player_color: Human player's color ('white' or 'black').
        """
        self.player_color = PlayerColor(player_color)
        
        # Core components
        self.frame_capture: Optional[FrameCapture] = None
        self.board_detector = BoardDetector()
        self.perspective = PerspectiveTransformer()
        self.square_extractor = SquareExtractor()
        self.piece_detector = PieceDetector()
        self.move_detector = MoveDetector()
        self.move_validator = MoveValidator()
        self.engine: Optional[StockfishEngine] = None
        
        # Game state
        self._state = GameState.INITIALIZING
        self._board_state = BoardState()
        self._previous_detection: Optional[np.ndarray] = None
        self._pending_ai_move: Optional[str] = None
        self._last_player_move: Optional[str] = None
        
        # Configuration
        self._stability_frames = config.game.get('stability_frames', 5)
        self._stable_count = 0
        
        # History
        self._move_history: List[str] = []
        self._events: List[GameEvent] = []
        
        # Callbacks
        self._on_state_change: Optional[Callable[[GameState], None]] = None
        self._on_move: Optional[Callable[[str, str], None]] = None  # (move, player)
        self._on_error: Optional[Callable[[str], None]] = None
    
    def initialize(self, camera_id: int = 0, engine_path: Optional[str] = None) -> bool:
        """Initialize all components.
        
        Args:
            camera_id: Camera device ID.
            engine_path: Path to Stockfish executable.
            
        Returns:
            True if initialization successful.
        """
        self._log_event('initialize', {'camera_id': camera_id})
        
        # Initialize camera
        self.frame_capture = FrameCapture(camera_id)
        if not self.frame_capture.start():
            logger.error("Failed to start camera")
            return False
        
        # Initialize engine
        self.engine = StockfishEngine(path=engine_path)
        if not self.engine.start():
            logger.warning("Failed to start engine - AI moves unavailable")
            # Continue without engine for testing
        
        self._set_state(GameState.CALIBRATING)
        logger.info("Game controller initialized")
        return True
    
    def shutdown(self) -> None:
        """Shutdown all components."""
        if self.frame_capture:
            self.frame_capture.stop()
        if self.engine:
            self.engine.quit()
        
        self._log_event('shutdown', {})
        logger.info("Game controller shut down")
    
    def calibrate_board(self, corners: Optional[np.ndarray] = None) -> bool:
        """Calibrate board detection.
        
        Args:
            corners: Manual corner coordinates, or None for auto-detect.
            
        Returns:
            True if calibration successful.
        """
        if self.frame_capture is None:
            return False
        
        frame = self.frame_capture.get_frame()
        if frame is None:
            logger.error("No frame available for calibration")
            return False
        
        if corners is not None:
            self.board_detector.calibrate(corners)
        else:
            result = self.board_detector.detect(frame)
            if not result.found:
                logger.warning("Board not detected - manual calibration required")
                return False
            self.board_detector.calibrate(result.corners)
        
        # Store initial board state from detection
        self._detect_current_state(frame)
        
        self._set_state(GameState.WAITING_PLAYER_MOVE if self._is_player_turn() 
                       else GameState.COMPUTING_AI_MOVE)
        
        self._log_event('calibrate', {'manual': corners is not None})
        return True
    
    def process_frame(self, frame: Optional[np.ndarray] = None) -> None:
        """Process a camera frame and update game state.
        
        Args:
            frame: BGR frame to process. If None, captures from camera.
        """
        if frame is None:
            if self.frame_capture is None:
                return
            frame = self.frame_capture.get_frame()
            if frame is None:
                return
        
        # Detect board
        detection = self.board_detector.detect(frame)
        if not detection.found:
            if self._state != GameState.BOARD_NOT_DETECTED:
                self._set_state(GameState.BOARD_NOT_DETECTED)
            return
        
        # Transform to top-down view
        board_image = self.perspective.transform(frame, detection.corners)
        if board_image is None:
            return
        
        # Extract squares
        squares = self.square_extractor.extract_all(board_image)
        
        # Detect pieces
        current_detection = self.piece_detector.classify_board(squares)
        
        # Process based on current state
        if self._state == GameState.BOARD_NOT_DETECTED:
            self._set_state(GameState.WAITING_PLAYER_MOVE if self._is_player_turn()
                          else GameState.WAITING_AI_EXECUTION)
        
        elif self._state == GameState.WAITING_PLAYER_MOVE:
            self._handle_waiting_player(current_detection)
        
        elif self._state == GameState.WAITING_AI_EXECUTION:
            self._handle_waiting_ai_execution(current_detection)
    
    def _handle_waiting_player(self, current_detection: np.ndarray) -> None:
        """Handle state when waiting for player move."""
        if self._previous_detection is None:
            self._previous_detection = current_detection
            return
        
        # Check if board changed
        if np.array_equal(current_detection, self._previous_detection):
            self._stable_count += 1
            return
        
        # Board changed - check stability
        self._stable_count = 0
        
        # Detect move
        detected = self.move_detector.detect_move(
            self._previous_detection,
            current_detection,
            self._board_state.board
        )
        
        if detected:
            # Validate move
            result = self.move_validator.validate(detected.uci, self._board_state)
            
            if result.is_valid:
                self._execute_player_move(result.move)
                self._previous_detection = current_detection
            else:
                self._handle_illegal_move(detected.uci, result.error)
        else:
            # Detection incomplete - wait for more frames
            self._previous_detection = current_detection
    
    def _handle_waiting_ai_execution(self, current_detection: np.ndarray) -> None:
        """Handle state when waiting for AI move execution."""
        if self._pending_ai_move is None:
            return
        
        if self._previous_detection is None:
            self._previous_detection = current_detection
            return
        
        # Check if board changed
        if np.array_equal(current_detection, self._previous_detection):
            return
        
        # Detect what move was made
        detected = self.move_detector.detect_move(
            self._previous_detection,
            current_detection,
            self._board_state.board
        )
        
        if detected and detected.uci == self._pending_ai_move:
            # Correct AI move executed
            self._board_state.make_move(self._pending_ai_move)
            self._move_history.append(self._pending_ai_move)
            self._log_event('ai_move_executed', {'move': self._pending_ai_move})
            
            self._pending_ai_move = None
            self._previous_detection = current_detection
            
            # Check game over
            if self._check_game_over():
                return
            
            self._set_state(GameState.WAITING_PLAYER_MOVE)
        else:
            # Wrong move or partial move - wait
            self._previous_detection = current_detection
    
    def _execute_player_move(self, move: str) -> None:
        """Execute a validated player move."""
        self._board_state.make_move(move)
        self._move_history.append(move)
        self._last_player_move = move
        
        if self._on_move:
            self._on_move(move, 'player')
        
        self._log_event('player_move', {'move': move})
        logger.info(f"Player move: {move}")
        
        # Check game over
        if self._check_game_over():
            return
        
        # Get AI response
        self._set_state(GameState.COMPUTING_AI_MOVE)
        self._compute_ai_move()
    
    def _compute_ai_move(self) -> None:
        """Compute and display AI's move."""
        if self.engine is None or not self.engine.is_ready:
            logger.warning("Engine not available")
            self._set_state(GameState.WAITING_PLAYER_MOVE)
            return
        
        self.engine.set_position(self._board_state.fen)
        ai_move = self.engine.get_best_move()
        
        if ai_move:
            self._pending_ai_move = ai_move
            self._log_event('ai_move_computed', {'move': ai_move})
            logger.info(f"AI move: {ai_move}")
            
            if self._on_move:
                self._on_move(ai_move, 'ai')
            
            self._set_state(GameState.SHOWING_AI_MOVE)
        else:
            logger.error("Engine failed to produce move")
            self._set_state(GameState.WAITING_PLAYER_MOVE)
    
    def confirm_ai_move(self) -> None:
        """Confirm AI move display and wait for execution."""
        if self._pending_ai_move:
            self._set_state(GameState.WAITING_AI_EXECUTION)
    
    def _handle_illegal_move(self, move: str, error: str) -> None:
        """Handle an illegal move detection."""
        logger.warning(f"Illegal move: {move} - {error}")
        
        if self._on_error:
            self._on_error(f"Illegal move: {move}")
        
        self._log_event('illegal_move', {'move': move, 'error': error})
        self._set_state(GameState.ILLEGAL_MOVE_DETECTED)
    
    def acknowledge_error(self) -> None:
        """Acknowledge an error and return to waiting."""
        if self._state in {GameState.ILLEGAL_MOVE_DETECTED, GameState.BOARD_NOT_DETECTED}:
            self._set_state(GameState.WAITING_PLAYER_MOVE if self._is_player_turn()
                          else GameState.WAITING_AI_EXECUTION)
    
    def _check_game_over(self) -> bool:
        """Check if game is over and update state."""
        if self._board_state.is_checkmate():
            winner = "Black" if self._board_state.turn == chess.WHITE else "White"
            self._log_event('game_over', {'result': 'checkmate', 'winner': winner})
            self._set_state(GameState.GAME_OVER_CHECKMATE)
            return True
        
        if self._board_state.is_stalemate():
            self._log_event('game_over', {'result': 'stalemate'})
            self._set_state(GameState.GAME_OVER_STALEMATE)
            return True
        
        if self._board_state.board.is_insufficient_material():
            self._log_event('game_over', {'result': 'draw'})
            self._set_state(GameState.GAME_OVER_DRAW)
            return True
        
        return False
    
    def _is_player_turn(self) -> bool:
        """Check if it's the human player's turn."""
        if self.player_color == PlayerColor.WHITE:
            return self._board_state.turn == chess.WHITE
        return self._board_state.turn == chess.BLACK
    
    def _detect_current_state(self, frame: np.ndarray) -> None:
        """Detect and store current board state from frame."""
        detection = self.board_detector.detect(frame)
        if detection.found:
            board_image = self.perspective.transform(frame, detection.corners)
            if board_image is not None:
                squares = self.square_extractor.extract_all(board_image)
                self._previous_detection = self.piece_detector.classify_board(squares)
    
    def _set_state(self, new_state: GameState) -> None:
        """Change game state and notify callbacks."""
        old_state = self._state
        self._state = new_state
        
        logger.debug(f"State: {old_state.name} -> {new_state.name}")
        
        if self._on_state_change:
            self._on_state_change(new_state)
    
    def _log_event(self, event_type: str, data: dict) -> None:
        """Log a game event."""
        event = GameEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data
        )
        self._events.append(event)
    
    # Public properties and methods
    
    @property
    def state(self) -> GameState:
        """Get current game state."""
        return self._state
    
    @property
    def board_state(self) -> BoardState:
        """Get current board state."""
        return self._board_state
    
    @property
    def pending_ai_move(self) -> Optional[str]:
        """Get pending AI move to be executed."""
        return self._pending_ai_move
    
    @property
    def move_history(self) -> List[str]:
        """Get move history."""
        return self._move_history.copy()
    
    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return is_game_over_state(self._state)
    
    def set_on_state_change(self, callback: Callable[[GameState], None]) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback
    
    def set_on_move(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for moves (move, player)."""
        self._on_move = callback
    
    def set_on_error(self, callback: Callable[[str], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback
    
    def new_game(self, player_color: str = 'white') -> None:
        """Start a new game."""
        self.player_color = PlayerColor(player_color)
        self._board_state = BoardState()
        self._previous_detection = None
        self._pending_ai_move = None
        self._move_history = []
        self._stable_count = 0
        
        self._log_event('new_game', {'player_color': player_color})
        
        if self.board_detector.is_calibrated:
            self._set_state(GameState.WAITING_PLAYER_MOVE if self._is_player_turn()
                          else GameState.COMPUTING_AI_MOVE)
        else:
            self._set_state(GameState.CALIBRATING)
    
    def export_pgn(self) -> str:
        """Export game to PGN format."""
        import chess.pgn
        
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Vision System Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        node = game
        board = chess.Board()
        
        for move_uci in self._move_history:
            move = chess.Move.from_uci(move_uci)
            node = node.add_variation(move)
            board.push(move)
        
        return str(game)
