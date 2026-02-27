"""Frame capture from camera using OpenCV."""
import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time

from src.config import config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FrameCapture:
    """Handles camera frame capture with threading support."""
    
    def __init__(self, camera_id: Optional[int] = None):
        """Initialize the frame capture.
        
        Args:
            camera_id: Camera device ID. If None, uses config value.
        """
        self.camera_id = camera_id if camera_id is not None else config.camera.get('device_id', 0)
        self.width = config.camera.get('width', 1280)
        self.height = config.camera.get('height', 720)
        self.fps = config.camera.get('fps', 30)
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_frame_time = 0.0
        
    def start(self) -> bool:
        """Start capturing frames.
        
        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._running:
            logger.warning("Capture already running")
            return True
            
        self._cap = cv2.VideoCapture(self.camera_id)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read actual properties (may differ from requested)
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop capturing frames."""
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            
        logger.info("Camera capture stopped")
    
    def _capture_loop(self) -> None:
        """Internal capture loop running in separate thread."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            
            if ret:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
            else:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame.
        
        Returns:
            BGR frame as numpy array, or None if no frame available.
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Wait for and return a new frame.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            BGR frame as numpy array, or None if timeout.
        """
        start_time = time.time()
        initial_count = self._frame_count
        
        while time.time() - start_time < timeout:
            if self._frame_count > initial_count:
                return self.get_frame()
            time.sleep(0.01)
        
        return None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame without starting continuous capture.
        
        Returns:
            BGR frame as numpy array, or None if capture failed.
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Read a few frames to let camera adjust
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    
    def set_resolution(self, width: int, height: int) -> None:
        """Set capture resolution.
        
        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.width = width
        self.height = height
        
        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Resolution set to {width}x{height}")
    
    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    @property
    def actual_fps(self) -> float:
        """Get actual FPS based on frame timing."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self) -> 'FrameCapture':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
