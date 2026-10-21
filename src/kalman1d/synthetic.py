"""Synthetic incubator trace with a known true temperature, for the tests and one figure.

The truth follows the same thermal model as the filter (heater lag included, plus a little
process noise so the model is not perfect), driven by an on/off thermostat with
hysteresis, like the real device. The reading is truth + Gaussian noise, rounded to the
sensor's 0.01 °C resolution, once every 15 s. This data is generated, never measured; it is
labelled synthetic wherever it appears.
"""

from __future__ import annotations

import numpy as np

from .data import Trace
from .model import ThermalModel


def make_synthetic(
    model: ThermalModel,
    duration_min: float = 240.0,
    dt_s: float = 15.0,
    setpoint: float = 35.0,
    hysteresis: float = 0.2,
    temp0: float = 27.0,
    ambient: float = 27.5,
    noise_std: float = 0.03,
    process_std: float = 0.01,
    resolution: float = 0.01,
    seed: int = 0,
) -> tuple[Trace, np.ndarray]:
    """Return ``(trace, truth)``; ``truth`` is the exact chamber temperature at each reading."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_min * 60 / dt_s)) + 1
    dt = dt_s / 60.0
    t_min = np.arange(n) * dt
    temp_out = ambient + 0.5 * np.sin(2 * np.pi * t_min / duration_min)
    truth, heater = np.empty(n), np.zeros(n)
    truth[0], on, heat = temp0, 0.0, 0.0
    for k in range(1, n):
        truth[k] = model.step(truth[k - 1], temp_out[k - 1], heat, dt) + process_std * rng.normal()
        heat = model.heat_step(heat, heater[k - 1], dt)
        if truth[k] < setpoint - hysteresis:
            on = 1.0
        elif truth[k] > setpoint + hysteresis:
            on = 0.0
        heater[k] = on
    temp = np.round((truth + noise_std * rng.normal(size=n)) / resolution) * resolution
    trace = Trace(t_min, temp, temp_out, heater, np.full(n, setpoint))
    return trace, truth
