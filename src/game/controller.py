"""Game controller – orchestrates camera, vision, state and engine.

Performance design
------------------
* Frame processing uses a **frame-skip counter** so the computer-vision
  pipeline runs at most once every ``process_every_n_frames`` camera frames.
  This reduces CPU load significantly on hardware where the camera can produce
  frames faster than the pipeline can process them.
* Board corners are cached inside ``BoardDetector``; no re-detection unless
  the user explicitly triggers re-calibration.
* Engine calls are fire-and-forget (synchronous for simplicity); the engine
  subprocess stays alive across moves to avoid process-spawn overhead.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import chess

from src.camera.capture import ThreadedCapture
from src.engine.stockfish_engine import StockfishEngine
from src.state.board_state import BoardState
from src.state.move_detector import MoveDetector
from src.vision.board_detector import BoardDetector
from src.vision.piece_detector import PieceDetector

logger = logging.getLogger(__name__)


class GameController:
    """Top-level orchestrator for the smart chessboard system."""

    def __init__(
        self,
        camera_device: int = 0,
        player_color: bool = chess.WHITE,
        engine_path: str = "stockfish",
        engine_skill: int = 10,
        engine_move_time: float = 1.0,
        process_every_n_frames: int = 3,
        enable_theatre: bool = True,
        debug: bool = False,
    ) -> None:
        self._player_color = player_color
        self._process_every_n = process_every_n_frames
        self._frame_counter = 0
        self._running = False

        # Camera
        self._capture = ThreadedCapture(device_id=camera_device)

        # Vision pipeline
        self._board_detector = BoardDetector()
        self._piece_detector = PieceDetector()

        # State
        self._chess_board = chess.Board()
        self._board_state = BoardState()
        self._move_detector = MoveDetector()

        # Engine
        self._engine = StockfishEngine(
            path=engine_path,
            skill_level=engine_skill,
            move_time=engine_move_time,
        )

        # Theatre (optional)
        self._theatre = None
        if enable_theatre:
            try:
                from src.theatre.display import Theatre
                self._theatre = Theatre()
            except Exception as exc:
                logger.warning("Theatre unavailable: %s", exc)

        logger.info("GameController initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main game loop."""
        self._running = True
        logger.info("Game loop started. Press Q to quit.")

        # Wait for initial board detection.
        self._calibrate()

        # Initialise move detector with starting state.
        self._move_detector.reset(self._board_state)

        while self._running:
            frame = self._capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Frame-skip: only run vision every N frames to save CPU.
            self._frame_counter += 1
            if self._frame_counter % self._process_every_n != 0:
                if self._theatre:
                    self._theatre.render()
                self._handle_input()
                continue

            warped = self._board_detector.detect(frame)
            if not warped.found or warped.warped is None:
                logger.debug("Board not visible – skipping frame.")
                self._handle_input()
                continue

            occupancy = self._piece_detector.detect(warped.warped)
            current_state = BoardState()
            current_state.update(occupancy)

            move = self._move_detector.push(current_state, self._chess_board)
            if move is not None:
                self._apply_player_move(move)

            if self._theatre:
                self._theatre.update(self._chess_board, status=self._status_text())
                self._theatre.render()

            self._handle_input()

        self._shutdown()

    def new_game(self) -> None:
        self._chess_board.reset()
        self._move_detector.reset(self._board_state)
        logger.info("New game started.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self) -> None:
        logger.info("Waiting for board detection…")
        self._board_detector.reset()
        while True:
            frame = self._capture.read()
            if frame is None:
                time.sleep(0.05)
                continue
            warped = self._board_detector.detect(frame)
            if warped.found and warped.warped is not None:
                occupancy = self._piece_detector.detect(warped.warped)
                self._board_state.update(occupancy)
                logger.info("Board calibrated.")
                return
            time.sleep(0.05)

    def _apply_player_move(self, move: chess.Move) -> None:
        if self._chess_board.turn != self._player_color:
            logger.warning("Not the player's turn – ignoring detected move %s.", move)
            return
        self._chess_board.push(move)
        logger.info("Player moved: %s | FEN: %s", move.uci(), self._chess_board.fen())

        if self._chess_board.is_game_over():
            logger.info("Game over: %s", self._chess_board.result())
            return

        # AI response.
        ai_uci = self._engine.get_best_move(self._chess_board)
        if ai_uci:
            ai_move = chess.Move.from_uci(ai_uci)
            self._chess_board.push(ai_move)
            logger.info("AI moved: %s", ai_uci)

    def _handle_input(self) -> None:
        if self._theatre is None:
            return
        keys = self._theatre.poll_events()
        if keys.get("q") or keys.get("quit"):
            self._running = False
        if keys.get("c"):
            self._calibrate()
        if keys.get("n"):
            self.new_game()
        if keys.get("f"):
            self._theatre.flip()

    def _status_text(self) -> str:
        if self._chess_board.is_checkmate():
            return "Checkmate!"
        if self._chess_board.is_stalemate():
            return "Stalemate"
        if self._chess_board.is_check():
            return "Check!"
        turn = "White" if self._chess_board.turn == chess.WHITE else "Black"
        return f"{turn} to move"

    def _shutdown(self) -> None:
        self._capture.release()
        self._engine.close()
        if self._theatre:
            self._theatre.close()
        logger.info("Shutdown complete.")
