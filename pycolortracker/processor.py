from math import floor
from typing import List, Tuple

import cv2
import numpy as np
from numba import jit, prange, uint8


class Processor:
    def __init__(self) -> None:
        self.hue = 0
        self.threshold = 20
        self.roi = (0, 0, -1, -1)

        self.bbox_data: List[Tuple[int, Tuple[int, int, int, int]]] = []

    def callback_process_data(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        frame_bgr_roi = frame_bgr[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        frame_hsv = cv2.cvtColor(frame_bgr_roi, cv2.COLOR_BGR2HSV)
        frame_color_intensity = self.process_hvs_frame_into_color_intensity(frame_hsv, self.hue)
        _, frame_color_intensity_binary = cv2.threshold(frame_color_intensity, self.threshold, 255, cv2.THRESH_BINARY)
        return cv2.boundingRect(frame_color_intensity_binary)

    def callback_process_time(self, time_: int, bbox: Tuple[int, int, int, int]):
        self.bbox_data.append((time_, bbox))

    @staticmethod
    @jit(uint8[:,:,::1](uint8[:,:,::1], uint8), nopython=True, parallel=True, nogil=True)
    def process_hvs_frame_into_color_intensity(frame: np.ndarray, hue: int) -> np.ndarray:
        height, width, _ = frame.shape
        output = np.zeros((height, width, 1), dtype=uint8)
        for y in prange(height):
            for x in prange(width):
                output[y, x, 0] = (255 - floor(abs(((frame[y, x, 0] + 90 - hue) % 180) - 90) * 2.833)) * frame[y, x, 1] / 255 * frame[y, x, 2] / 255
        return output
