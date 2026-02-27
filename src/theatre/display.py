"""Visual "theatre" display using Pygame.

Performance design
------------------
* Board squares and piece sprites are rendered to a cached ``Surface`` and
  only redrawn when the game state changes (dirty-flag pattern).
* Font objects are created once and reused; ``pygame.font.Font()`` is
  expensive to call repeatedly.
* The display loop caps frame rate via ``pygame.time.Clock.tick()`` to avoid
  busy-waiting / wasted GPU cycles.
"""

from __future__ import annotations

import logging
from typing import Optional

import chess
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYGAME_AVAILABLE = False
    logger.warning("Pygame not installed – Theatre display disabled.")


# Colour palette
_LIGHT_SQ = (240, 217, 181)
_DARK_SQ = (181, 136, 99)
_HIGHLIGHT = (246, 246, 105, 180)
_WHITE_PIECE = (255, 255, 255)
_BLACK_PIECE = (30, 30, 30)
_TEXT_COLOUR = (50, 50, 50)
_BG_COLOUR = (40, 40, 40)


class Theatre:
    """Pygame window that renders the current chess game state.

    Parameters
    ----------
    window_size:
        Total window size in pixels.
    board_size:
        Board area in pixels (centred in the window).
    fps:
        Target rendering frame rate.
    flipped:
        When *True* the board is rendered from Black's perspective.
    """

    def __init__(
        self,
        window_size: int = 800,
        board_size: int = 640,
        fps: int = 30,
        flipped: bool = False,
    ) -> None:
        if not _PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for Theatre display.")

        pygame.init()
        self._screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Chess-Evo")
        self._clock = pygame.time.Clock()
        self._fps = fps
        self._sq_px = board_size // 8
        self._board_offset = (window_size - board_size) // 2
        self._flipped = flipped

        # Pre-render the static board surface (only rebuilt on flip).
        self._board_surf: pygame.Surface = self._make_board_surface(board_size)
        self._dirty = True

        # Fonts – created once and cached.
        self._font_sm = pygame.font.SysFont("monospace", 14)
        self._font_md = pygame.font.SysFont("monospace", 20, bold=True)

        # Current game state.
        self._chess_board: Optional[chess.Board] = None
        self._last_move: Optional[chess.Move] = None
        self._status_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        status: str = "",
    ) -> None:
        """Update displayed state. Does NOT draw immediately (lazy rendering)."""
        self._chess_board = board
        self._last_move = last_move
        self._status_text = status
        self._dirty = True

    def flip(self) -> None:
        """Flip the board orientation and rebuild the cached board surface."""
        self._flipped = not self._flipped
        board_size = self._sq_px * 8
        self._board_surf = self._make_board_surface(board_size)
        self._dirty = True

    def render(self) -> None:
        """Draw one frame; respects the FPS cap and dirty flag."""
        if self._dirty:
            self._draw()
            self._dirty = False
        self._clock.tick(self._fps)

    def poll_events(self) -> dict[str, bool]:
        """Process pygame events and return a dict of key flags."""
        keys: dict[str, bool] = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keys["quit"] = True
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key).lower()
                keys[key_name] = True
        return keys

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._screen.fill(_BG_COLOUR)
        o = self._board_offset
        self._screen.blit(self._board_surf, (o, o))

        if self._chess_board is not None:
            self._draw_highlights(o)
            self._draw_pieces(o)

        if self._status_text:
            surf = self._font_md.render(self._status_text, True, (220, 220, 220))
            self._screen.blit(surf, (10, 10))

        pygame.display.flip()

    def _draw_highlights(self, offset: int) -> None:
        if self._last_move is None:
            return
        highlight_surf = pygame.Surface((self._sq_px, self._sq_px), pygame.SRCALPHA)
        highlight_surf.fill(_HIGHLIGHT)
        for sq in (self._last_move.from_square, self._last_move.to_square):
            col, row = self._sq_to_screen(sq)
            self._screen.blit(highlight_surf, (offset + col, offset + row))

    def _draw_pieces(self, offset: int) -> None:
        assert self._chess_board is not None
        for sq in chess.SQUARES:
            piece = self._chess_board.piece_at(sq)
            if piece is None:
                continue
            col, row = self._sq_to_screen(sq)
            colour = _WHITE_PIECE if piece.color == chess.WHITE else _BLACK_PIECE
            symbol = piece.unicode_symbol()
            surf = self._font_md.render(symbol, True, colour)
            cx = offset + col + (self._sq_px - surf.get_width()) // 2
            cy = offset + row + (self._sq_px - surf.get_height()) // 2
            self._screen.blit(surf, (cx, cy))

    def _sq_to_screen(self, sq: int) -> tuple[int, int]:
        """Return (pixel_col, pixel_row) for the top-left of square *sq*."""
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if self._flipped:
            col = (7 - file) * self._sq_px
            row = rank * self._sq_px
        else:
            col = file * self._sq_px
            row = (7 - rank) * self._sq_px
        return col, row

    def _make_board_surface(self, board_size: int) -> "pygame.Surface":
        sq_px = board_size // 8
        surf = pygame.Surface((board_size, board_size))
        for rank in range(8):
            for file in range(8):
                colour = _LIGHT_SQ if (rank + file) % 2 == 0 else _DARK_SQ
                rect = pygame.Rect(file * sq_px, rank * sq_px, sq_px, sq_px)
                surf.fill(colour, rect)
        return surf
