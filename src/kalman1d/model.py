"""Thermal model of the chamber. It is the Kalman filter's process model.

    dE/dt = (a·u − E) / τh          heat reaching the chamber lags the relay u (0/1)
    dT/dt = E − b·(T − T_out)        chamber warms with E, loses heat to ambient at rate b

``T`` chamber temperature (°C), ``E`` effective heating rate (°C/min), ``a`` heating rate with
the heater on (°C/min), ``τh`` heater lag (min), ``b`` loss rate (1/min, 1/b is the thermal
time constant). ``E`` is computed from the known relay history alone, so the Kalman state
stays one-dimensional: only ``T`` is estimated. The numbers come from data/model_params.json
(``second_order`` block), fitted in incubator-telemetry-analysis on the same day of data that
this repo filters; that fit has an open-loop error of 0.16 °C RMS over the day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalModel:
    a: float  # °C/min, chamber heating rate with the heater on (after the lag settles)
    b: float  # 1/min, heat-loss rate towards ambient
    tau_h: float  # min, first-order lag between the relay and the heat felt by the chamber

    @classmethod
    def from_params(cls, params: dict) -> ThermalModel:
        """Build from the ``second_order`` block of model_params.json."""
        return cls(
            float(params["a_c_per_min"]), float(params["b_per_min"]), float(params["tau_h_min"])
        )

    def heat_step(self, heat: float, heater: float, dt_min: float) -> float:
        """Exact update of the lagged heat input over one step with the relay held at ``heater``."""
        target = self.a * heater
        return target + (heat - target) * np.exp(-dt_min / self.tau_h)

    def heat_input(self, heater: np.ndarray, t_min: np.ndarray, heat0: float = 0.0) -> np.ndarray:
        """Heat felt by the chamber at each reading, from the relay history (cold start default)."""
        heat = np.empty_like(t_min)
        heat[0] = heat0
        for k in range(1, len(t_min)):
            heat[k] = self.heat_step(heat[k - 1], heater[k - 1], t_min[k] - t_min[k - 1])
        return heat

    def step(self, temp: float, temp_out: float, heat: float, dt_min: float) -> float:
        """Forward-Euler step of the chamber temperature over ``dt_min`` minutes."""
        return temp + dt_min * (heat - self.b * (temp - temp_out))

    def transition(self, dt_min: float) -> float:
        """d(T_next)/d(T): the scalar state-transition factor ``F`` for one step."""
        return 1.0 - self.b * dt_min

    def simulate(
        self, temp0: float, temp_out: np.ndarray, heater: np.ndarray, t_min: np.ndarray
    ) -> np.ndarray:
        """Open-loop simulation from ``temp0``: what the model alone predicts, no sensor."""
        heat = self.heat_input(heater, t_min)
        temp = np.empty_like(t_min)
        temp[0] = temp0
        for k in range(1, len(t_min)):
            temp[k] = self.step(temp[k - 1], temp_out[k - 1], heat[k - 1], t_min[k] - t_min[k - 1])
        return temp
