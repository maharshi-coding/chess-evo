"""Threaded camera capture.

Performance design
------------------
* A dedicated background thread continuously grabs frames from the camera so
  the main processing loop never blocks waiting for V4L2 / DirectShow I/O.
* Only the *latest* frame is kept (old frames are discarded), so downstream
  code always works with fresh data without building up a queue.
* ``process_every_n_frames`` further reduces CPU load by allowing the caller
  to skip frames cheaply.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCapture:
    """Non-blocking camera capture backed by a reader thread.

    Parameters
    ----------
    device_id:
        OpenCV camera device index.
    width, height:
        Requested capture resolution (camera may override).
    fps:
        Requested frame rate.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")

        # Request resolution / FPS – actual values depend on hardware.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Use MJPEG when available for lower USB bandwidth and higher throughput.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(
            "ThreadedCapture started – device=%d  res=%dx%d  fps=%d",
            device_id,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._cap.get(cv2.CAP_PROP_FPS)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (or *None* if none yet)."""
        with self._lock:
            return self._frame if self._frame is None else self._frame.copy()

    def release(self) -> None:
        """Stop the reader thread and release the camera."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info("ThreadedCapture released.")

    def __enter__(self) -> "ThreadedCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal reader loop (background thread)
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Continuously grab frames; only store the most recent one.

        ``cv2.VideoCapture.grab()`` is used instead of ``read()`` to pull
        frames through the driver buffer without decoding them.  ``retrieve()``
        is called immediately afterwards to decode only the frame we actually
        intend to use.  This pattern keeps the driver buffer from filling up,
        which would otherwise cause latency to build up over time.
        """
        while not self._stop_event.is_set():
            grabbed = self._cap.grab()
            if not grabbed:
                logger.warning("Camera grab failed – retrying in 50 ms.")
                time.sleep(0.05)
                continue

            ok, frame = self._cap.retrieve()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
