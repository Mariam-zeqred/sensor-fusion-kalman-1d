"""Load the incubator history JSON (copied from incubator-telemetry-analysis) into numpy arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORY_FILE = DATA_DIR / "hive-1C5A00_history_2026-10-03.json"
MODEL_PARAMS_FILE = DATA_DIR / "model_params.json"


@dataclass(frozen=True)
class Trace:
    """One telemetry trace. All arrays have the same length; time is minutes from the first row.

    ``temp`` may contain NaN where the chamber reading is missing (the filter skips the update
    there). The ambient reading and heater state are assumed always present: they are the
    model's inputs, not measurements of the state.
    """

    t_min: np.ndarray  # minutes since the first reading
    temp: np.ndarray  # chamber SHT31, °C (NaN = no reading)
    temp_out: np.ndarray  # ambient SHT31, °C
    heater: np.ndarray  # relay state, 0.0 / 1.0
    target: np.ndarray  # setpoint, °C

    def __len__(self) -> int:
        return len(self.t_min)

    @property
    def dt_min(self) -> np.ndarray:
        """Step length before each reading; the first step is set to the median step."""
        dt = np.diff(self.t_min)
        return np.concatenate(([np.median(dt)], dt))

    def window(self, start_min: float, stop_min: float) -> np.ndarray:
        """Boolean mask of the readings with ``start_min <= t < stop_min``."""
        return (self.t_min >= start_min) & (self.t_min < stop_min)

    def with_dropout(self, start_min: float, stop_min: float) -> Trace:
        """Copy of the trace with the chamber reading removed in ``[start_min, stop_min)``."""
        temp = self.temp.copy()
        temp[self.window(start_min, stop_min)] = np.nan
        return replace(self, temp=temp)


def load_history(path: str | Path = HISTORY_FILE) -> Trace:
    """Read the device's history export (a JSON list of readings) sorted by time."""
    rows = sorted(json.loads(Path(path).read_text()), key=lambda r: r["ts"])
    ts = np.array([r["ts"] for r in rows], dtype=float)
    return Trace(
        t_min=(ts - ts[0]) / 60.0,
        temp=np.array([r["temp"] for r in rows], dtype=float),
        temp_out=np.array([r["tempOut"] for r in rows], dtype=float),
        heater=np.array([1.0 if r["heater"] else 0.0 for r in rows]),
        target=np.array([r["targetTemp"] for r in rows], dtype=float),
    )


def load_model_params(path: str | Path = MODEL_PARAMS_FILE) -> dict:
    """The ``second_order`` block (heater-lag model) of the source repo's fitted thermal model."""
    return json.loads(Path(path).read_text())["second_order"]
