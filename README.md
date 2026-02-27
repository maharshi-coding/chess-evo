# Camera-Based Smart Chessboard System

A vision-based smart chessboard that uses a camera to detect moves made on a physical chessboard and responds with AI moves via Stockfish.

## Features

- Real-time board detection using computer vision
- Automatic move detection by comparing board states
- Move validation using chess rules
- AI opponent powered by Stockfish
- Visual "theatre" display showing the game state
- Support for all chess moves including castling, en passant, and promotion

## Requirements

- Python 3.8 or higher
- Webcam or external camera
- Physical chessboard with clear contrast
- Standard chess pieces (Staunton recommended)
- Stockfish chess engine (optional, for AI)

## Installation

### 1. Install Python

Download and install Python from [python.org](https://www.python.org/downloads/)

During installation, make sure to:
- Check "Add Python to PATH"
- Check "Install pip"

### 2. Clone the Repository

```bash
git clone <repository-url>
cd chess-evo
```

### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Stockfish (Optional)

Download Stockfish from [stockfishchess.org](https://stockfishchess.org/download/)

Extract to `stockfish/` folder or update `config.yaml` with the path.

## Usage

### Basic Usage

```bash
python -m src.main
```

### Command Line Options

```bash
python -m src.main --help

Options:
  --camera INT    Camera device ID (default: 0)
  --color STR     Player color: white/black (default: white)
  --engine STR    Path to Stockfish executable
  --skill INT     Engine skill level 0-20 (default: 10)
  --no-theatre    Disable visual display
  --debug         Enable debug logging
```

### Controls

During the game:
- **C** - Calibrate board (re-detect corners)
- **N** - New game
- **F** - Flip board orientation
- **Space** - Confirm AI move (proceed to execution)
- **Escape** - Acknowledge error and continue
- **Q** - Quit

## Calibration

1. Position the camera to see the entire chessboard
2. Ensure good lighting without harsh shadows
3. Press **C** to calibrate
4. The system will detect the board corners automatically

For best results:
- Use a board with good contrast (green/cream or brown/cream)
- Avoid reflective surfaces
- Keep the camera stable
- Ensure all pieces are clearly visible

## Project Structure

```
chess-evo/
├── src/
│   ├── camera/         # Camera capture
│   ├── vision/         # Board and piece detection
│   ├── state/          # Board state and move detection
│   ├── engine/         # Stockfish integration
│   ├── game/           # Game controller
│   ├── theatre/        # Visual display
│   └── utils/          # Utilities
├── tests/              # Test suite
├── docs/               # Documentation
├── tasks/              # Project tracking
├── config.yaml         # Configuration
└── requirements.txt    # Dependencies
```

## Configuration

Edit `config.yaml` to customize:

```yaml
camera:
  device_id: 0          # Camera ID
  width: 1280           # Resolution
  height: 720

vision:
  board_detection:
    method: "contour"   # Detection method
  piece_detection:
    method: "color"     # Color-based detection

engine:
  path: "stockfish/stockfish.exe"
  skill_level: 10       # 0-20
  move_time: 1.0        # Seconds per move

theatre:
  window_size: 800
  board_size: 640
```

## Troubleshooting

### Board Not Detected

- Ensure entire board is visible in frame
- Improve lighting conditions
- Adjust camera angle
- Try manual calibration by clicking corners

### Pieces Not Detected Correctly

- Ensure pieces have good contrast with board
- Adjust detection thresholds in config
- Clean the camera lens

### Engine Not Working

- Verify Stockfish path in config
- Download latest Stockfish version
- Check engine has execute permissions

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) for chess logic
- [Stockfish](https://stockfishchess.org/) chess engine
- [OpenCV](https://opencv.org/) for computer vision
- [Pygame](https://pygame.org/) for display
