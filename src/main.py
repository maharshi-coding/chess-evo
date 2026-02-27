"""Entry point – ``python -m src.main``."""

from __future__ import annotations

import argparse
import sys

import chess

from src.utils.config import load_config
from src.utils.logger import setup_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-based smart chessboard system."
    )
    parser.add_argument("--camera", type=int, default=None, metavar="INT",
                        help="Camera device ID (default: from config)")
    parser.add_argument("--color", type=str, default=None,
                        choices=["white", "black"],
                        help="Player colour (default: white)")
    parser.add_argument("--engine", type=str, default=None, metavar="PATH",
                        help="Path to Stockfish executable")
    parser.add_argument("--skill", type=int, default=None, metavar="INT",
                        help="Engine skill level 0-20 (default: from config)")
    parser.add_argument("--no-theatre", action="store_true",
                        help="Disable visual display")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()

    log_level = "DEBUG" if args.debug else cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("logging", {}).get("file")
    setup_logging(level=log_level, log_file=log_file)

    # Resolve configuration (CLI args take precedence over config.yaml).
    camera_device = args.camera if args.camera is not None else cfg.get("camera", {}).get("device_id", 0)
    player_color = chess.BLACK if (args.color or "white") == "black" else chess.WHITE
    engine_path = args.engine or cfg.get("engine", {}).get("path", "stockfish")
    skill = args.skill if args.skill is not None else cfg.get("engine", {}).get("skill_level", 10)
    move_time = float(cfg.get("engine", {}).get("move_time", 1.0))
    process_n = int(cfg.get("camera", {}).get("process_every_n_frames", 3))
    enable_theatre = not args.no_theatre

    from src.game.controller import GameController

    controller = GameController(
        camera_device=camera_device,
        player_color=player_color,
        engine_path=engine_path,
        engine_skill=skill,
        engine_move_time=move_time,
        process_every_n_frames=process_n,
        enable_theatre=enable_theatre,
        debug=args.debug,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
