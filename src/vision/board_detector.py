"""Board detection using computer vision techniques."""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.config import config
from src.utils.logging_setup import get_logger
from src.utils.helpers import order_points

logger = get_logger(__name__)


@dataclass
class BoardDetectionResult:
    """Result of board detection."""
    found: bool
    corners: Optional[np.ndarray] = None  # 4 corners in order: TL, TR, BR, BL
    confidence: float = 0.0
    debug_image: Optional[np.ndarray] = None


class BoardDetector:
    """Detects chessboard in camera frames."""
    
    def __init__(self):
        """Initialize the board detector."""
        vision_config = config.vision.get('board_detection', {})
        self.method = vision_config.get('method', 'contour')
        self.min_board_area = vision_config.get('min_board_area', 50000)
        self.canny_threshold1 = vision_config.get('canny_threshold1', 50)
        self.canny_threshold2 = vision_config.get('canny_threshold2', 150)
        
        # Calibrated corners (set by calibration routine)
        self._calibrated_corners: Optional[np.ndarray] = None
        self._is_calibrated = False
        
    def detect(self, frame: np.ndarray, debug: bool = False) -> BoardDetectionResult:
        """Detect chessboard in the given frame.
        
        Args:
            frame: BGR image from camera.
            debug: If True, include debug visualization in result.
            
        Returns:
            BoardDetectionResult with corner positions if found.
        """
        if self._is_calibrated and self._calibrated_corners is not None:
            # Use calibrated corners directly
            return BoardDetectionResult(
                found=True,
                corners=self._calibrated_corners.copy(),
                confidence=1.0
            )
        
        if self.method == 'contour':
            return self._detect_by_contour(frame, debug)
        elif self.method == 'checkerboard':
            return self._detect_checkerboard(frame, debug)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return BoardDetectionResult(found=False)
    
    def _detect_by_contour(self, frame: np.ndarray, debug: bool = False) -> BoardDetectionResult:
        """Detect board by finding the largest quadrilateral contour."""
        debug_image = frame.copy() if debug else None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.debug("No contours found")
            return BoardDetectionResult(found=False, debug_image=debug_image)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest contours
            area = cv2.contourArea(contour)
            
            if area < self.min_board_area:
                continue
            
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                ordered_corners = order_points(corners)
                
                # Calculate confidence based on how square-like the shape is
                confidence = self._calculate_board_confidence(ordered_corners)
                
                if debug and debug_image is not None:
                    cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 3)
                    for i, corner in enumerate(ordered_corners):
                        cv2.circle(debug_image, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
                        cv2.putText(debug_image, str(i), tuple(corner.astype(int)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                logger.debug(f"Board detected with confidence {confidence:.2f}")
                return BoardDetectionResult(
                    found=True,
                    corners=ordered_corners,
                    confidence=confidence,
                    debug_image=debug_image
                )
        
        logger.debug("No valid quadrilateral found")
        return BoardDetectionResult(found=False, debug_image=debug_image)
    
    def _detect_checkerboard(self, frame: np.ndarray, debug: bool = False) -> BoardDetectionResult:
        """Detect board using OpenCV's checkerboard pattern detection."""
        debug_image = frame.copy() if debug else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find inner corners (7x7 for 8x8 board)
        pattern_size = (7, 7)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Get outer corners of the board
            inner_corners = corners.reshape(7, 7, 2)
            
            # Extrapolate to get board corners
            # This is an approximation - assumes regular grid
            top_left = self._extrapolate_corner(inner_corners[0, 0], inner_corners[0, 1], inner_corners[1, 0])
            top_right = self._extrapolate_corner(inner_corners[0, 6], inner_corners[0, 5], inner_corners[1, 6])
            bottom_right = self._extrapolate_corner(inner_corners[6, 6], inner_corners[6, 5], inner_corners[5, 6])
            bottom_left = self._extrapolate_corner(inner_corners[6, 0], inner_corners[6, 1], inner_corners[5, 0])
            
            board_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            
            if debug and debug_image is not None:
                cv2.drawChessboardCorners(debug_image, pattern_size, corners, ret)
                for corner in board_corners:
                    cv2.circle(debug_image, tuple(corner.astype(int)), 10, (0, 0, 255), -1)
            
            return BoardDetectionResult(
                found=True,
                corners=board_corners,
                confidence=0.95,
                debug_image=debug_image
            )
        
        return BoardDetectionResult(found=False, debug_image=debug_image)
    
    def _extrapolate_corner(self, corner: np.ndarray, adj1: np.ndarray, adj2: np.ndarray) -> np.ndarray:
        """Extrapolate outer corner from inner corners."""
        # Vector from adjacent corners to the corner
        v1 = corner - adj1
        v2 = corner - adj2
        # Extrapolate by adding both vectors
        return corner + v1 + v2
    
    def _calculate_board_confidence(self, corners: np.ndarray) -> float:
        """Calculate confidence score based on shape regularity."""
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        
        # Check aspect ratio (should be close to 1:1)
        avg_side = np.mean(sides)
        side_variance = np.std(sides) / avg_side if avg_side > 0 else 1.0
        
        # Calculate angles (should be close to 90 degrees)
        angles = []
        for i in range(4):
            p1 = corners[(i - 1) % 4]
            p2 = corners[i]
            p3 = corners[(i + 1) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(np.abs(angle - np.pi/2))
        
        angle_error = np.mean(angles)
        
        # Combine metrics
        confidence = max(0, 1.0 - side_variance - angle_error)
        return confidence
    
    def calibrate(self, corners: np.ndarray) -> None:
        """Set calibrated corners manually.
        
        Args:
            corners: 4 corner points in order (TL, TR, BR, BL).
        """
        if corners.shape != (4, 2):
            raise ValueError("Corners must be shape (4, 2)")
        
        self._calibrated_corners = order_points(corners.astype(np.float32))
        self._is_calibrated = True
        logger.info("Board calibration set")
    
    def reset_calibration(self) -> None:
        """Reset calibration to use automatic detection."""
        self._calibrated_corners = None
        self._is_calibrated = False
        logger.info("Board calibration reset")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if board is calibrated."""
        return self._is_calibrated
    
    @property
    def calibrated_corners(self) -> Optional[np.ndarray]:
        """Get calibrated corners if available."""
        return self._calibrated_corners.copy() if self._calibrated_corners is not None else None
