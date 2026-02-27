"""Main entry point for the Chess Vision system."""
import argparse
import sys
import time

import pygame

from src.utils.logging_setup import setup_logging, get_logger
from src.game import GameController, GameState
from src.theatre import TheatreRenderer
from src.state import BoardState

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Camera-Based Smart Chessboard System")
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--color', type=str, default='white',
                       choices=['white', 'black'],
                       help='Player color (default: white)')
    parser.add_argument('--engine', type=str, default=None,
                       help='Path to Stockfish executable')
    parser.add_argument('--skill', type=int, default=10,
                       help='Engine skill level 0-20 (default: 10)')
    parser.add_argument('--no-theatre', action='store_true',
                       help='Disable visual theatre display')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging()
    
    logger.info("Starting Chess Vision System")
    logger.info(f"Player color: {args.color}")
    
    # Initialize game controller
    game = GameController(player_color=args.color)
    
    # Initialize theatre display
    theatre = None
    if not args.no_theatre:
        theatre = TheatreRenderer()
        if not theatre.initialize():
            logger.warning("Failed to initialize theatre - running headless")
            theatre = None
    
    # Initialize game
    if not game.initialize(camera_id=args.camera, engine_path=args.engine):
        logger.error("Failed to initialize game controller")
        if theatre:
            theatre.shutdown()
        return 1
    
    # Set up callbacks
    def on_state_change(state: GameState):
        logger.info(f"Game state: {state.name}")
        if theatre:
            status_map = {
                GameState.CALIBRATING: "Position board in view for calibration",
                GameState.WAITING_PLAYER_MOVE: "Your turn - make a move",
                GameState.COMPUTING_AI_MOVE: "Computing AI move...",
                GameState.SHOWING_AI_MOVE: f"AI plays: {game.pending_ai_move or ''}",
                GameState.WAITING_AI_EXECUTION: "Execute AI move on board",
                GameState.BOARD_NOT_DETECTED: "Board not detected - reposition",
                GameState.ILLEGAL_MOVE_DETECTED: "Illegal move! Reset pieces",
                GameState.GAME_OVER_CHECKMATE: "Checkmate! Game over",
                GameState.GAME_OVER_STALEMATE: "Stalemate! Game drawn",
            }
            theatre.set_status(status_map.get(state, state.name))
    
    def on_move(move: str, player: str):
        logger.info(f"{player.capitalize()} move: {move}")
        if theatre:
            from_sq, to_sq = move[:2], move[2:4]
            if player == 'player':
                theatre.set_last_move(from_sq, to_sq)
                theatre.clear_ai_suggestion()
            else:  # AI move
                theatre.set_ai_suggestion(from_sq, to_sq)
    
    def on_error(error: str):
        logger.error(f"Game error: {error}")
    
    game.set_on_state_change(on_state_change)
    game.set_on_move(on_move)
    game.set_on_error(on_error)
    
    # Main loop
    running = True
    clock = pygame.time.Clock() if theatre else None
    
    logger.info("Entering main loop - Press 'C' to calibrate, 'Q' to quit")
    
    try:
        while running:
            # Process Pygame events
            if theatre:
                events = theatre.process_events()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_c:
                            # Manual calibration trigger
                            game.calibrate_board()
                        elif event.key == pygame.K_n:
                            # New game
                            game.new_game(args.color)
                        elif event.key == pygame.K_f:
                            # Flip board
                            theatre.flip_board()
                        elif event.key == pygame.K_SPACE:
                            # Confirm AI move
                            if game.state == GameState.SHOWING_AI_MOVE:
                                game.confirm_ai_move()
                        elif event.key == pygame.K_ESCAPE:
                            # Acknowledge error
                            game.acknowledge_error()
            
            # Process camera frame
            game.process_frame()
            
            # Render theatre
            if theatre:
                theatre.render(game.board_state)
                clock.tick(30)  # Limit to 30 FPS
            else:
                time.sleep(0.033)  # ~30 FPS equivalent
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        game.shutdown()
        if theatre:
            theatre.shutdown()
    
    # Export game if any moves were made
    if game.move_history:
        pgn = game.export_pgn()
        logger.info(f"Game PGN:\n{pgn}")
    
    logger.info("Chess Vision System stopped")
    return 0


if __name__ == '__main__':
    sys.exit(main())
