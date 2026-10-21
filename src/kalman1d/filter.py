"""Scalar Kalman filter: thermal model as the prediction, the chamber sensor as the update.

Per reading k (step length dt, E = heat input computed from the relay history, see model.py):

    predict   x⁻ = x + dt·(E − b·(x − T_out))        P⁻ = F²·P + q²·dt,   F = 1 − b·dt
    update    K  = P⁻ / (P⁻ + r²)                     (skipped when the reading is missing)
              x  = x⁻ + K·(z − x⁻)                     P  = (1 − K)·P⁻

``q`` is the process-noise standard deviation in °C per √minute (how much the real chamber
can drift away from the model per unit time), ``r`` the measurement-noise standard deviation
in °C. The ambient sensor and the relay state enter only through the prediction: they are
inputs that drive the chamber, not measurements of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Trace
from .model import ThermalModel


@dataclass(frozen=True)
class KalmanResult:
    estimate: np.ndarray  # posterior mean, °C
    variance: np.ndarray  # posterior variance, °C²
    predicted: np.ndarray  # prior mean (model step from the previous estimate), °C
    innovation: np.ndarray  # z − prior, NaN where the reading was missing
    innovation_var: np.ndarray  # S = P⁻ + r²
    gain: np.ndarray  # K, 0 where the reading was missing

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def nis(self) -> float:
        """Mean normalised innovation squared over the readings present (≈ 1 if consistent)."""
        ok = ~np.isnan(self.innovation)
        return float(np.mean(self.innovation[ok] ** 2 / self.innovation_var[ok]))

    def log_likelihood(self) -> float:
        """Gaussian log-likelihood of the innovations; the tuning objective for ``q``."""
        ok = ~np.isnan(self.innovation)
        s, nu = self.innovation_var[ok], self.innovation[ok]
        return float(-0.5 * np.sum(np.log(2 * np.pi * s) + nu**2 / s))


@dataclass(frozen=True)
class KalmanFilter1D:
    model: ThermalModel
    q: float  # process noise std, °C/√min
    r: float  # measurement noise std, °C

    def run(self, trace: Trace, x0: float | None = None, p0: float | None = None) -> KalmanResult:
        n = len(trace)
        heat = self.model.heat_input(trace.heater, trace.t_min)
        est, var, pred = np.empty(n), np.empty(n), np.empty(n)
        innov, s_arr, gain = np.full(n, np.nan), np.empty(n), np.zeros(n)
        first = int(np.argmax(~np.isnan(trace.temp)))
        x = float(trace.temp[first]) if x0 is None else x0
        p = self.r**2 if p0 is None else p0
        rr = self.r**2
        for k in range(n):
            if k > 0:
                dt = trace.t_min[k] - trace.t_min[k - 1]
                x = self.model.step(x, trace.temp_out[k - 1], heat[k - 1], dt)
                p = self.model.transition(dt) ** 2 * p + self.q**2 * dt
            pred[k] = x
            s = p + rr
            z = trace.temp[k]
            if not np.isnan(z):
                kk = p / s
                innov[k], gain[k] = z - x, kk
                x = x + kk * (z - x)
                p = (1.0 - kk) * p
            est[k], var[k], s_arr[k] = x, p, s
        return KalmanResult(est, var, pred, innov, s_arr, gain)

    def steady_state(self, dt_min: float) -> tuple[float, float, float]:
        """(prior variance, posterior variance, gain) the filter settles to on a regular grid.

        Positive root of the scalar Riccati equation P⁻ = F²·P⁻·R/(P⁻+R) + Q.
        """
        f2, qq, rr = self.model.transition(dt_min) ** 2, self.q**2 * dt_min, self.r**2
        bcoef = rr - f2 * rr - qq
        p_prior = 0.5 * (-bcoef + np.sqrt(bcoef**2 + 4 * qq * rr))
        k = p_prior / (p_prior + rr)
        return float(p_prior), float((1 - k) * p_prior), float(k)


def estimate_measurement_std(temp: np.ndarray) -> float:
    """Sensor noise from the second difference of the readings.

    For a slowly varying signal plus white noise of variance R, the second difference
    z[k+1] − 2z[k] + z[k−1] has variance 6R. The median absolute deviation (×1.4826) is
    used instead of the standard deviation so that gaps and relay kinks do not inflate it.
    """
    z = temp[~np.isnan(temp)]
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


def tune_process_noise(
    trace: Trace, model: ThermalModel, r: float, grid: np.ndarray
) -> tuple[float, np.ndarray]:
    """Pick ``q`` from ``grid`` by maximum innovation likelihood; returns (best q, all scores)."""
    scores = np.array([KalmanFilter1D(model, q, r).run(trace).log_likelihood() for q in grid])
    return float(grid[int(np.argmax(scores))]), scores
