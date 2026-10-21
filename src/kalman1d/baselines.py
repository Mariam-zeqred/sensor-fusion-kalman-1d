"""The two things the filter is compared against: the raw reading and a causal moving average."""

from __future__ import annotations

import numpy as np


def moving_average(temp: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` readings (fewer at the start); NaN readings are left out.

    Causal, so it can run on the device, and it lags a ramp by about half the window.
    """
    out = np.full(len(temp), np.nan)
    for k in range(len(temp)):
        chunk = temp[max(0, k - window + 1) : k + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[k] = chunk.mean()
    return out


def hold_last(temp: np.ndarray) -> np.ndarray:
    """Forward-fill: what a controller sees during a dropout if nothing smarter is done."""
    out = temp.copy()
    for k in range(1, len(out)):
        if np.isnan(out[k]):
            out[k] = out[k - 1]
    return out
