"""1-D Kalman filter fusing an incubator's chamber sensor, ambient sensor and heater state."""

from .baselines import hold_last, moving_average
from .data import Trace, load_history, load_model_params
from .filter import KalmanFilter1D, KalmanResult, estimate_measurement_std, tune_process_noise
from .model import ThermalModel
from .synthetic import make_synthetic

__all__ = [
    "KalmanFilter1D",
    "KalmanResult",
    "ThermalModel",
    "Trace",
    "estimate_measurement_std",
    "hold_last",
    "load_history",
    "load_model_params",
    "make_synthetic",
    "moving_average",
    "tune_process_noise",
]
