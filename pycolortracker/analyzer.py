import enum
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


class PlotType(enum.IntEnum):
    TIME = enum.auto()
    TIME_DELTA = enum.auto()
    POSITION_DELTA = enum.auto()
    VELOCITY = enum.auto()
    VELOCITY_DELTA = enum.auto()
    ACCELERATION = enum.auto()


class DataType(enum.IntEnum):
    TIME = enum.auto()
    TIME_DELTA = enum.auto()
    POSITION = enum.auto()
    POSITION_DELTA = enum.auto()
    VELOCITY = enum.auto()
    VELOCITY_DELTA = enum.auto()
    ACCELERATION = enum.auto()


class Analyzer:
    linewidth = 1.0
    
    def __init__(self, pixels_per_unit: float, unit: str, *args: PlotType) -> None:
        self.pixels_per_unit = pixels_per_unit
        self.unit = unit
        self.plot_types = args
        self.plot_functions = {
            PlotType.TIME: self.plot_time,
            PlotType.TIME_DELTA: self.plot_time_delta,
            PlotType.POSITION_DELTA: self.plot_position_delta,
            PlotType.VELOCITY: self.plot_velocity,
            PlotType.VELOCITY_DELTA: self.plot_velocity_delta,
            PlotType.ACCELERATION: self.plot_acceleration
        }
    
    def prepare_data(self, data: List[Tuple[int, Tuple[int, int, int, int]]]) -> Dict[DataType, np.ndarray]:
        none_filtered_data = list(filter(lambda record: not record[1] is None, data))
        time_ = np.array(list(map(lambda record: record[0], none_filtered_data)), dtype="float64") / 1e9 # ns to s
        time_delta = time_[1:] - time_[:-1]
        bbox = np.array(list(map(lambda record: record[1], none_filtered_data)), dtype="float64")
        position = np.swapaxes(np.array([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2]), 0, 1) / self.pixels_per_unit
        position_delta = position[1:] - position[:-1]
        position_delta = np.linalg.norm(position_delta, axis=1)
        velocity = position_delta / time_delta
        velocity = gaussian_filter1d(velocity, sigma=10) # lower sigma -> closer to original curve, higher sigma -> smoother curve
        velocity_delta = velocity[1:] - velocity[:-1]
        acceleration = velocity_delta / time_delta[:-1]
        acceleration = gaussian_filter1d(acceleration, sigma=10)
        return {
            DataType.TIME: time_,
            DataType.TIME_DELTA: time_delta,
            DataType.POSITION: position,
            DataType.POSITION_DELTA: position_delta,
            DataType.VELOCITY: velocity,
            DataType.VELOCITY_DELTA: velocity_delta,
            DataType.ACCELERATION: acceleration
        }

    def plot_data(self, data: List[Tuple[int, Tuple[int, int, int, int]]]) -> None:
        data_pool = self.prepare_data(data)
        nrows = min(len(self.plot_types), 3)
        ncols = -(len(self.plot_types) // -3) # ceil division
        plt.figure(figsize=(12, 9))
        iplot = 1
        for plot_type in self.plot_types:
            plt.subplot(nrows, ncols, iplot)
            self.plot_functions[plot_type](data_pool)
            iplot += 1
        plt.tight_layout()
        plt.savefig("figure.png", dpi=400)
        plt.show()
    
    def plot_time(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("Zeit t")
        plt.xlabel("Bild")
        plt.ylabel(f"t in s")
        plt.plot(data_pool[DataType.TIME], linewidth=self.linewidth)
    
    def plot_time_delta(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("ΔZeit Δt")
        plt.xlabel("Bild")
        plt.ylabel(f"Δt in s")
        plt.plot(data_pool[DataType.TIME_DELTA], linewidth=self.linewidth)
    
    def plot_position_delta(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("ΔPosition Δs(t)")
        plt.xlabel("t in s")
        plt.ylabel(f"Δs in {self.unit}")
        plt.plot(data_pool[DataType.TIME][:-1], data_pool[DataType.POSITION_DELTA], linewidth=self.linewidth)
    
    def plot_velocity(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("Geschwindigkeit v(t)")
        plt.xlabel("t in s")
        plt.ylabel(f"v in {self.unit}/s")
        plt.plot(data_pool[DataType.TIME][:-1], data_pool[DataType.VELOCITY], linewidth=self.linewidth)
    
    def plot_velocity_delta(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("ΔGeschwindigkeit Δv(t)")
        plt.xlabel("t in s")
        plt.ylabel(f"Δv in {self.unit}/s")
        plt.plot(data_pool[DataType.TIME][:-2], data_pool[DataType.VELOCITY_DELTA], linewidth=self.linewidth)

    def plot_acceleration(self, data_pool: Dict[DataType, np.ndarray]) -> None:
        plt.title("Beschleunigung a(t)")
        plt.xlabel("t in s")
        plt.ylabel(f"a in {self.unit}/s²")
        plt.plot(data_pool[DataType.TIME][:-2], data_pool[DataType.ACCELERATION], linewidth=self.linewidth)
