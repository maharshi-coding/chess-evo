"""Chess board rendering using Pygame."""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from typing import Optional, Tuple, List, Dict
import chess

from src.config import config
from src.utils.logging_setup import get_logger
from src.state import BoardState

logger = get_logger(__name__)


# Piece Unicode characters
PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}


class TheatreRenderer:
    """Renders the chess board using Pygame."""
    
    def __init__(self, window_size: int = 800, board_size: int = 640):
        """Initialize the renderer.
        
        Args:
            window_size: Total window size in pixels.
            board_size: Board rendering size in pixels.
        """
        # Get from config or use defaults
        theatre_config = config.theatre
        self.window_size = theatre_config.get('window_size', window_size)
        self.board_size = theatre_config.get('board_size', board_size)
        self.square_size = self.board_size // 8
        
        # Colors
        colors = theatre_config.get('colors', {})
        self.light_color = tuple(colors.get('light_square', [238, 238, 210]))
        self.dark_color = tuple(colors.get('dark_square', [118, 150, 86]))
        self.highlight_color = tuple(colors.get('highlight', [255, 255, 0, 128]))
        self.last_move_color = tuple(colors.get('last_move', [255, 255, 0, 100]))
        self.ai_move_color = tuple(colors.get('ai_move', [100, 255, 100, 150]))
        
        # Pygame state
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._piece_font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._is_initialized = False
        
        # Display state
        self._last_move: Optional[Tuple[str, str]] = None
        self._ai_suggestion: Optional[Tuple[str, str]] = None
        self._highlighted_squares: List[str] = []
        self._status_text: str = ""
        self._flipped = False  # True if black at bottom
        
    def initialize(self) -> bool:
        """Initialize Pygame and create window.
        
        Returns:
            True if initialization successful.
        """
        try:
            pygame.init()
            pygame.display.set_caption("Chess Vision - Theatre")
            
            self._screen = pygame.display.set_mode((self.window_size, self.window_size))
            self._font = pygame.font.SysFont('Arial', 24)
            self._piece_font = pygame.font.SysFont('Segoe UI Symbol', self.square_size - 10)
            self._small_font = pygame.font.SysFont('Arial', 16)
            
            self._is_initialized = True
            logger.info("Theatre renderer initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pygame: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown Pygame."""
        if self._is_initialized:
            pygame.quit()
            self._is_initialized = False
            logger.info("Theatre renderer shut down")
    
    def render(self, board_state: BoardState) -> None:
        """Render the chess board.
        
        Args:
            board_state: Current board state to render.
        """
        if not self._is_initialized or self._screen is None:
            return
        
        # Clear screen
        self._screen.fill((40, 40, 40))  # Dark background
        
        # Calculate board offset to center it
        board_offset_x = (self.window_size - self.board_size) // 2
        board_offset_y = 60  # Leave space for status at top
        
        # Draw board
        self._draw_board(board_offset_x, board_offset_y)
        
        # Draw highlights
        self._draw_highlights(board_offset_x, board_offset_y)
        
        # Draw pieces
        self._draw_pieces(board_state, board_offset_x, board_offset_y)
        
        # Draw coordinates
        self._draw_coordinates(board_offset_x, board_offset_y)
        
        # Draw status
        self._draw_status()
        
        # Update display
        pygame.display.flip()
    
    def _draw_board(self, offset_x: int, offset_y: int) -> None:
        """Draw the chess board squares."""
        for row in range(8):
            for col in range(8):
                x = offset_x + col * self.square_size
                y = offset_y + row * self.square_size
                
                is_light = (row + col) % 2 == 0
                color = self.light_color if is_light else self.dark_color
                
                pygame.draw.rect(self._screen, color, 
                               (x, y, self.square_size, self.square_size))
    
    def _draw_highlights(self, offset_x: int, offset_y: int) -> None:
        """Draw square highlights."""
        # Last move highlight
        if self._last_move:
            self._highlight_square(self._last_move[0], self.last_move_color, 
                                  offset_x, offset_y)
            self._highlight_square(self._last_move[1], self.last_move_color,
                                  offset_x, offset_y)
        
        # AI suggestion highlight
        if self._ai_suggestion:
            self._highlight_square(self._ai_suggestion[0], self.ai_move_color,
                                  offset_x, offset_y)
            self._highlight_square(self._ai_suggestion[1], self.ai_move_color,
                                  offset_x, offset_y)
            # Draw arrow
            self._draw_arrow(self._ai_suggestion[0], self._ai_suggestion[1],
                           offset_x, offset_y)
        
        # Custom highlights
        for sq in self._highlighted_squares:
            self._highlight_square(sq, self.highlight_color, offset_x, offset_y)
    
    def _highlight_square(self, square: str, color: Tuple, 
                         offset_x: int, offset_y: int) -> None:
        """Highlight a single square with transparency."""
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        
        if self._flipped:
            col = 7 - col
            row = 7 - row
        
        x = offset_x + col * self.square_size
        y = offset_y + row * self.square_size
        
        # Create transparent surface
        highlight_surface = pygame.Surface((self.square_size, self.square_size))
        highlight_surface.set_alpha(color[3] if len(color) > 3 else 128)
        highlight_surface.fill(color[:3])
        
        self._screen.blit(highlight_surface, (x, y))
    
    def _draw_arrow(self, from_sq: str, to_sq: str, 
                   offset_x: int, offset_y: int) -> None:
        """Draw an arrow between two squares."""
        from_col = ord(from_sq[0]) - ord('a')
        from_row = 8 - int(from_sq[1])
        to_col = ord(to_sq[0]) - ord('a')
        to_row = 8 - int(to_sq[1])
        
        if self._flipped:
            from_col, from_row = 7 - from_col, 7 - from_row
            to_col, to_row = 7 - to_col, 7 - to_row
        
        start_x = offset_x + from_col * self.square_size + self.square_size // 2
        start_y = offset_y + from_row * self.square_size + self.square_size // 2
        end_x = offset_x + to_col * self.square_size + self.square_size // 2
        end_y = offset_y + to_row * self.square_size + self.square_size // 2
        
        # Draw arrow line
        pygame.draw.line(self._screen, (0, 150, 0), 
                        (start_x, start_y), (end_x, end_y), 4)
        
        # Draw arrowhead (simple circle at end for now)
        pygame.draw.circle(self._screen, (0, 150, 0), (end_x, end_y), 8)
    
    def _draw_pieces(self, board_state: BoardState, 
                    offset_x: int, offset_y: int) -> None:
        """Draw chess pieces on the board."""
        if self._piece_font is None:
            return
        
        for row in range(8):
            for col in range(8):
                # Get square name
                if self._flipped:
                    sq_col = 7 - col
                    sq_row = row
                else:
                    sq_col = col
                    sq_row = 7 - row
                
                square = chr(ord('a') + sq_col) + str(sq_row + 1)
                piece = board_state.get_piece_at(square)
                
                if piece:
                    x = offset_x + col * self.square_size + self.square_size // 2
                    y = offset_y + row * self.square_size + self.square_size // 2
                    
                    unicode_char = PIECE_UNICODE.get(piece, '?')
                    
                    # Render piece with shadow effect
                    text = self._piece_font.render(unicode_char, True, (0, 0, 0))
                    shadow_rect = text.get_rect(center=(x + 2, y + 2))
                    self._screen.blit(text, shadow_rect)
                    
                    # Main piece
                    color = (255, 255, 255) if piece.isupper() else (60, 60, 60)
                    text = self._piece_font.render(unicode_char, True, color)
                    text_rect = text.get_rect(center=(x, y))
                    self._screen.blit(text, text_rect)
    
    def _draw_coordinates(self, offset_x: int, offset_y: int) -> None:
        """Draw board coordinates."""
        if self._small_font is None:
            return
        
        files = 'abcdefgh'
        ranks = '12345678'
        
        if self._flipped:
            files = files[::-1]
            ranks = ranks[::-1]
        
        # Draw file letters (a-h)
        for i, letter in enumerate(files):
            x = offset_x + i * self.square_size + self.square_size // 2
            y = offset_y + self.board_size + 5
            
            text = self._small_font.render(letter, True, (200, 200, 200))
            rect = text.get_rect(center=(x, y + 10))
            self._screen.blit(text, rect)
        
        # Draw rank numbers (1-8)
        for i, number in enumerate(ranks[::-1]):
            x = offset_x - 15
            y = offset_y + i * self.square_size + self.square_size // 2
            
            text = self._small_font.render(number, True, (200, 200, 200))
            rect = text.get_rect(center=(x, y))
            self._screen.blit(text, rect)
    
    def _draw_status(self) -> None:
        """Draw status text at top of screen."""
        if self._font is None:
            return
        
        text = self._font.render(self._status_text, True, (255, 255, 255))
        rect = text.get_rect(center=(self.window_size // 2, 30))
        self._screen.blit(text, rect)
    
    def set_last_move(self, from_sq: str, to_sq: str) -> None:
        """Set the last move to highlight."""
        self._last_move = (from_sq, to_sq)
    
    def clear_last_move(self) -> None:
        """Clear last move highlight."""
        self._last_move = None
    
    def set_ai_suggestion(self, from_sq: str, to_sq: str) -> None:
        """Set AI move suggestion to display."""
        self._ai_suggestion = (from_sq, to_sq)
    
    def clear_ai_suggestion(self) -> None:
        """Clear AI suggestion."""
        self._ai_suggestion = None
    
    def highlight_squares(self, squares: List[str]) -> None:
        """Highlight specific squares."""
        self._highlighted_squares = squares.copy()
    
    def clear_highlights(self) -> None:
        """Clear all highlights."""
        self._highlighted_squares = []
    
    def set_status(self, text: str) -> None:
        """Set status text."""
        self._status_text = text
    
    def flip_board(self) -> None:
        """Toggle board orientation."""
        self._flipped = not self._flipped
    
    def set_orientation(self, black_at_bottom: bool) -> None:
        """Set board orientation."""
        self._flipped = black_at_bottom
    
    def process_events(self) -> List[pygame.event.Event]:
        """Process and return Pygame events."""
        events = []
        for event in pygame.event.get():
            events.append(event)
            if event.type == pygame.QUIT:
                return events
        return events
    
    def get_square_at_pos(self, x: int, y: int) -> Optional[str]:
        """Get square name at screen position."""
        board_offset_x = (self.window_size - self.board_size) // 2
        board_offset_y = 60
        
        # Check if within board
        if (x < board_offset_x or x >= board_offset_x + self.board_size or
            y < board_offset_y or y >= board_offset_y + self.board_size):
            return None
        
        col = (x - board_offset_x) // self.square_size
        row = (y - board_offset_y) // self.square_size
        
        if self._flipped:
            col = 7 - col
            row = 7 - row
        else:
            row = 7 - row
        
        return chr(ord('a') + col) + str(row + 1)
    
    @property
    def is_initialized(self) -> bool:
        """Check if renderer is initialized."""
        return self._is_initialized
