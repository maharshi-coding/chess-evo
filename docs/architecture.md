# Architecture Document - Camera-Based Smart Chessboard

## System Overview

This document describes the technical architecture for a vision-based smart chessboard system that uses a camera to detect moves made on a physical chessboard and responds with AI moves via a chess engine.

---

## High-Level Architecture

```
                              ┌─────────────────────────────────────┐
                              │         PHYSICAL WORLD              │
                              │                                     │
                              │    ┌─────────────────────────┐     │
                              │    │     Physical Board      │     │
                              │    │    (User makes moves)   │     │
                              │    └───────────┬─────────────┘     │
                              │                │                    │
                              │         ┌──────▼──────┐            │
                              │         │   Camera    │            │
                              │         └──────┬──────┘            │
                              └────────────────┼───────────────────┘
                                               │
┌──────────────────────────────────────────────┼───────────────────────────────┐
│                              SOFTWARE SYSTEM │                               │
│                                              │                               │
│  ┌───────────────────────────────────────────▼────────────────────────────┐ │
│  │                         VISION LAYER                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │   Frame      │─▶│    Board     │─▶│   Square     │─▶│   Piece    │ │ │
│  │  │   Capture    │  │   Detector   │  │  Extractor   │  │  Detector  │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬─────┘ │ │
│  └───────────────────────────────────────────────────────────────┼───────┘ │
│                                                                   │         │
│  ┌───────────────────────────────────────────────────────────────▼───────┐ │
│  │                         LOGIC LAYER                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │    Board     │─▶│    Move      │─▶│    Move      │─▶│   Game     │ │ │
│  │  │    State     │  │   Detector   │  │  Validator   │  │ Controller │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬─────┘ │ │
│  └───────────────────────────────────────────────────────────────┼───────┘ │
│                                                                   │         │
│  ┌──────────────────────────────────────────────────────────────┬┼───────┐ │
│  │                         ENGINE LAYER                         ││        │ │
│  │  ┌──────────────────────────────────────────────────────┐   ││        │ │
│  │  │                    Stockfish Engine                   │◀──┘│        │ │
│  │  │               (UCI Protocol Interface)                │────┘        │ │
│  │  └──────────────────────────────────────────────────────┘              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                   │         │
│  ┌───────────────────────────────────────────────────────────────▼───────┐ │
│  │                       PRESENTATION LAYER                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │ │
│  │  │   Board      │  │    Move      │  │   Status     │                 │ │
│  │  │  Renderer    │  │ Highlighter  │  │    Panel     │                 │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Vision Layer

#### 1.1 Frame Capture
```python
# Responsibility: Capture frames from camera
# Input: Camera device ID or stream URL
# Output: BGR frames (numpy array)
# Frequency: 30 FPS (configurable)

class FrameCapture:
    def __init__(self, camera_id: int = 0)
    def start() -> None
    def stop() -> None
    def get_frame() -> np.ndarray
    def set_resolution(width: int, height: int) -> None
```

#### 1.2 Board Detector
```python
# Responsibility: Find chessboard in frame
# Input: BGR frame
# Output: 4 corner points of the board
# Method: OpenCV findChessboardCorners or contour detection

class BoardDetector:
    def __init__(self)
    def detect(frame: np.ndarray) -> Optional[np.ndarray]  # Returns 4 corners
    def calibrate(frame: np.ndarray) -> bool  # Interactive calibration
    def is_calibrated() -> bool
```

#### 1.3 Square Extractor
```python
# Responsibility: Extract 64 individual squares
# Input: Frame + 4 corners
# Output: Dict mapping square name to image patch

class SquareExtractor:
    def __init__(self)
    def extract(frame: np.ndarray, corners: np.ndarray) -> Dict[str, np.ndarray]
    def get_perspective_transform(corners: np.ndarray) -> np.ndarray
```

#### 1.4 Piece Detector
```python
# Responsibility: Classify each square
# Input: 64 square images
# Output: Board state (8x8 matrix)
# Values: 0=empty, 1=white, -1=black (or piece types if ML model)

class PieceDetector:
    def __init__(self)
    def classify_square(square_img: np.ndarray) -> int
    def classify_board(squares: Dict[str, np.ndarray]) -> np.ndarray
    def set_thresholds(empty_thresh: float, white_thresh: float) -> None
```

---

### 2. Logic Layer

#### 2.1 Board State
```python
# Responsibility: Track current board state
# Input: Detection results
# Output: FEN string, piece positions

class BoardState:
    def __init__(self, fen: str = STARTING_FEN)
    def update_from_detection(detection: np.ndarray) -> None
    def get_fen() -> str
    def get_piece_at(square: str) -> Optional[str]
    def set_piece_at(square: str, piece: Optional[str]) -> None
    def copy() -> BoardState
```

#### 2.2 Move Detector
```python
# Responsibility: Detect moves from state changes
# Input: Previous state, current state
# Output: Move in UCI format (e.g., "e2e4")

class MoveDetector:
    def __init__(self)
    def detect_move(prev: BoardState, curr: BoardState) -> Optional[str]
    def get_changed_squares(prev: BoardState, curr: BoardState) -> List[str]
    def is_capture(move: str, prev: BoardState) -> bool
    def is_castling(move: str, prev: BoardState) -> bool
```

#### 2.3 Move Validator
```python
# Responsibility: Validate moves against chess rules
# Input: Move string, current position
# Output: Boolean (valid/invalid)

class MoveValidator:
    def __init__(self)
    def is_legal(move: str, board_state: BoardState) -> bool
    def get_legal_moves(board_state: BoardState) -> List[str]
    def get_move_type(move: str) -> MoveType  # NORMAL, CAPTURE, CASTLE, EN_PASSANT, PROMOTION
```

#### 2.4 Game Controller
```python
# Responsibility: Main game loop orchestration
# Input: Events from all layers
# Output: Game state updates

class GameController:
    def __init__(self)
    def start_game(player_color: str = 'white') -> None
    def process_frame(frame: np.ndarray) -> None
    def get_game_state() -> GameState  # WAITING_PLAYER, WAITING_AI, GAME_OVER
    def get_pending_ai_move() -> Optional[str]
    def handle_illegal_move(detected_move: str) -> None
    def reset_game() -> None
```

---

### 3. Engine Layer

#### 3.1 Stockfish Interface
```python
# Responsibility: Communicate with Stockfish
# Input: FEN position
# Output: Best move

class StockfishEngine:
    def __init__(self, path: str, skill_level: int = 10)
    def set_position(fen: str) -> None
    def get_best_move(time_limit: float = 1.0) -> str
    def get_evaluation() -> float  # Centipawns
    def set_skill_level(level: int) -> None  # 0-20
    def quit() -> None
```

---

### 4. Presentation Layer

#### 4.1 Board Renderer
```python
# Responsibility: Draw the chess board
# Input: Board state
# Output: Pygame surface

class BoardRenderer:
    def __init__(self, size: int = 640)
    def render(board_state: BoardState) -> pygame.Surface
    def highlight_square(square: str, color: Tuple[int, int, int]) -> None
    def highlight_move(from_sq: str, to_sq: str) -> None
    def show_ai_suggestion(move: str) -> None
```

---

## State Machine

```
                        ┌──────────────────┐
                        │      START       │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   CALIBRATING    │◀─────────────────┐
                        │  (Detect board)  │                  │
                        └────────┬─────────┘                  │
                                 │ Board found                │
                                 ▼                            │
                        ┌──────────────────┐                  │
               ┌───────▶│ WAITING_PLAYER   │                  │
               │        │  (User's turn)   │                  │
               │        └────────┬─────────┘                  │
               │                 │ Move detected              │
               │                 ▼                            │
               │        ┌──────────────────┐                  │
               │        │  VALIDATING      │──────────────────┤
               │        │                  │ Board lost       │
               │        └────────┬─────────┘                  │
               │                 │ Valid move                 │
               │                 ▼                            │
               │        ┌──────────────────┐                  │
               │        │  WAITING_AI      │                  │
               │        │  (AI's turn)     │                  │
               │        └────────┬─────────┘                  │
               │                 │ AI move calculated         │
               │                 ▼                            │
               │        ┌──────────────────┐                  │
               │        │ SHOWING_AI_MOVE  │                  │
               │        │ (User executes)  │                  │
               │        └────────┬─────────┘                  │
               │                 │ AI move detected on board  │
               └─────────────────┘                            │
                                                              │
                        ┌──────────────────┐                  │
                        │   GAME_OVER      │                  │
                        │  (Checkmate/Draw)│──────────────────┘
                        └──────────────────┘    New game
```

---

## Directory Structure

```
chess-evo/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration management
│   │
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── capture.py          # Frame capture
│   │   └── calibration.py      # Camera calibration
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── board_detector.py   # Find board in frame
│   │   ├── perspective.py      # Perspective transform
│   │   ├── square_extractor.py # Extract 64 squares
│   │   └── piece_detector.py   # Classify squares
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   ├── board_state.py      # Board representation
│   │   ├── move_detector.py    # Detect moves
│   │   └── move_validator.py   # Validate moves
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   └── stockfish.py        # Stockfish interface
│   │
│   ├── game/
│   │   ├── __init__.py
│   │   ├── controller.py       # Game orchestration
│   │   ├── states.py           # State machine
│   │   └── history.py          # Move history
│   │
│   ├── theatre/
│   │   ├── __init__.py
│   │   ├── renderer.py         # Board rendering
│   │   ├── pieces.py           # Piece sprites
│   │   └── ui.py               # UI components
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # Logging setup
│       └── helpers.py          # Utility functions
│
├── assets/
│   └── pieces/                 # Piece images for theatre
│
├── tests/
│   ├── __init__.py
│   ├── test_vision.py
│   ├── test_state.py
│   ├── test_engine.py
│   └── fixtures/               # Test images
│
├── tasks/
│   ├── todo.md                 # Project tasks
│   └── lessons.md              # Lessons learned
│
├── logs/                       # Runtime logs
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## Configuration Schema

```yaml
# config.yaml
camera:
  device_id: 0
  width: 1280
  height: 720
  fps: 30

vision:
  board_detection:
    method: "contour"  # or "checkerboard"
    min_board_area: 50000
  
  piece_detection:
    method: "color"  # or "ml"
    empty_threshold: 0.3
    white_threshold: 0.6

engine:
  path: "stockfish/stockfish.exe"
  skill_level: 10
  move_time: 1.0

theatre:
  window_size: 800
  board_size: 640
  colors:
    light_square: [238, 238, 210]
    dark_square: [118, 150, 86]
    highlight: [255, 255, 0]
    ai_move: [100, 255, 100]

game:
  player_color: "white"
  time_control: null  # null for unlimited

logging:
  level: "INFO"
  save_frames: false
  frame_directory: "logs/frames"
```

---

## Error Handling Strategy

| Error | Detection | Recovery |
|-------|-----------|----------|
| Board not in frame | No corners detected | Show "Position board in view" |
| Board partially obscured | <4 corners detected | Show "Entire board must be visible" |
| Ambiguous detection | Multiple valid moves possible | Request frame stabilization |
| Illegal move detected | Validator rejects | Show error, wait for correction |
| Engine timeout | No response in 5s | Retry or fallback |
| Camera disconnect | Read fails | Attempt reconnect |

---

## Testing Strategy

### Unit Tests
- Vision components with fixture images
- Board state operations
- Move detection logic
- Validator against python-chess

### Integration Tests  
- Vision pipeline end-to-end
- Full game flow simulation
- Engine communication

### Manual Tests
- Different lighting conditions
- Different board/piece sets
- All special moves
- Error recovery scenarios

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame processing | <100ms per frame |
| Move detection latency | <500ms after move complete |
| Engine response | <2s for move |
| UI refresh rate | 30 FPS |
| Memory usage | <500MB |

---

## Dependencies

```
opencv-python>=4.8.0
python-chess>=1.9.0
pygame>=2.5.0
numpy>=1.24.0
pyyaml>=6.0
```

Optional for ML-based detection:
```
torch>=2.0.0
torchvision>=0.15.0
```
