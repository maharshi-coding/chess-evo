"""Game state definitions."""
from enum import Enum, auto


class GameState(Enum):
    """States in the game state machine."""
    
    # Initialization states
    INITIALIZING = auto()
    CALIBRATING = auto()
    
    # Active game states
    WAITING_PLAYER_MOVE = auto()
    DETECTING_PLAYER_MOVE = auto()
    VALIDATING_MOVE = auto()
    PROCESSING_PLAYER_MOVE = auto()
    
    COMPUTING_AI_MOVE = auto()
    SHOWING_AI_MOVE = auto()
    WAITING_AI_EXECUTION = auto()
    VALIDATING_AI_EXECUTION = auto()
    
    # Error states
    BOARD_NOT_DETECTED = auto()
    ILLEGAL_MOVE_DETECTED = auto()
    WAITING_CORRECTION = auto()
    
    # End states
    GAME_OVER_CHECKMATE = auto()
    GAME_OVER_STALEMATE = auto()
    GAME_OVER_DRAW = auto()
    GAME_OVER_RESIGNATION = auto()
    
    # Paused
    PAUSED = auto()


class PlayerColor(Enum):
    """Player color selection."""
    WHITE = "white"
    BLACK = "black"


def state_requires_camera(state: GameState) -> bool:
    """Check if a state requires active camera monitoring."""
    return state in {
        GameState.CALIBRATING,
        GameState.WAITING_PLAYER_MOVE,
        GameState.DETECTING_PLAYER_MOVE,
        GameState.SHOWING_AI_MOVE,
        GameState.WAITING_AI_EXECUTION,
        GameState.VALIDATING_AI_EXECUTION,
        GameState.WAITING_CORRECTION
    }


def is_game_over_state(state: GameState) -> bool:
    """Check if state represents game over."""
    return state in {
        GameState.GAME_OVER_CHECKMATE,
        GameState.GAME_OVER_STALEMATE,
        GameState.GAME_OVER_DRAW,
        GameState.GAME_OVER_RESIGNATION
    }
