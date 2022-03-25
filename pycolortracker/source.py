import time
from typing import Any, Callable, Union, overload

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class ThreadedSource(QThread):
    frame_available = pyqtSignal(QImage)

    @overload
    def __init__(self, camera_selector: Union[int, str]) -> None:
        ...
    
    @overload
    def __init__(self, reusable_source: "ThreadedSource") -> None:
        ...

    def __init__(self, camera_source) -> None:
        super().__init__()
        if isinstance(camera_source, ThreadedSource):
            self.video_capture = camera_source.video_capture
        else:
            if isinstance(camera_source, str) and camera_source.isdigit():
                camera_source = int(camera_source)
            self.video_capture = cv2.VideoCapture(camera_source)
        if not self.video_capture.isOpened():
            raise IOError("Could not open VideoCapture.")
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.callback_process_data: Callable[[np.ndarray], Any] = None
        self.callback_process_time: Callable[[int, Any], None] = None
        self.callback_process_user_image: Callable[[np.ndarray, Any], np.ndarray] = None
        self.frame_available_timeout = 40_000000
        self.release_video_capture = True

    def run(self) -> None:
        start_time = time.perf_counter_ns()
        next_frame_available_time = 0
        while not self.isInterruptionRequested():
            success, cv_image = self.video_capture.read()
            if not success:
                break
            
            callback_data = self.callback_process_data(cv_image) if self.callback_process_data else None
            time_ = time.perf_counter_ns() - start_time  # time_ = time since start of capturing (in ns)
            if self.callback_process_time:
                self.callback_process_time(time_, callback_data)

            if time_ >= next_frame_available_time:
                next_frame_available_time = time_ + self.frame_available_timeout
                if self.callback_process_user_image:
                    cv_image = self.callback_process_user_image(cv_image, callback_data)
                qt_image = QImage(cv_image.data, cv_image.shape[1], cv_image.shape[0], QImage.Format.Format_BGR888)
                self.frame_available.emit(qt_image)
        if self.release_video_capture:
            self.video_capture.release()

    def stop_gracefully(self, timeout: int = 500) -> None:
        self.requestInterruption()
        if self.wait(timeout):
            self.terminate()
    
    def reuse(self) -> "ThreadedSource":
        self.release_video_capture = False
        self.stop_gracefully()
        return ThreadedSource(self)
    
    def release(self) -> None:
        self.video_capture.release()
