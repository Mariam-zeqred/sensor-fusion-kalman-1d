"""Run the filter on the real day, make the figures and write figures/results.json.

Every number quoted in the README comes from this file's output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .baselines import hold_last, moving_average  # noqa: E402
from .data import Trace, load_history, load_model_params  # noqa: E402
from .filter import (  # noqa: E402
    KalmanFilter1D,
    KalmanResult,
    estimate_measurement_std,
    tune_process_noise,
)
from .model import ThermalModel  # noqa: E402
from .synthetic import make_synthetic  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
MA_WINDOW = 8  # readings = 2 min at 15 s
Q_GRID = np.round(np.geomspace(0.005, 0.5, 41), 5)  # °C/√min
ZOOM = (226.0, 246.0)  # min from start: steady state at 38.8 °C with one heater pulse
DROPOUTS = {"ramp": (190.0, 200.0), "steady": (250.0, 260.0)}  # min from start, 10 min each

C_RAW, C_MA, C_KF, C_AMB, C_HEAT, C_TRUTH = (
    "#6b6b66",
    "#eb6834",
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#111",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 9,
        }
    )


def _heater_shading(ax: plt.Axes, trace: Trace, mask: np.ndarray | None = None) -> None:
    t, u = trace.t_min, trace.heater
    if mask is not None:
        t, u = t[mask], u[mask]
    ax.fill_between(
        t,
        0,
        1,
        where=u > 0.5,
        transform=ax.get_xaxis_transform(),
        color=C_HEAT,
        alpha=0.15,
        step="post",
        linewidth=0,
        label="heater on",
    )


def _band(ax: plt.Axes, t: np.ndarray, res: KalmanResult, mask: np.ndarray) -> None:
    ax.fill_between(
        t[mask],
        (res.estimate - 2 * res.std)[mask],
        (res.estimate + 2 * res.std)[mask],
        color=C_KF,
        alpha=0.2,
        linewidth=0,
        label="Kalman ±2σ",
    )


def fig_overview(trace: Trace, res: KalmanResult, ma: np.ndarray, out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    allm = np.ones(len(trace), bool)
    _heater_shading(ax, trace)
    ax.plot(
        trace.t_min,
        trace.target,
        color=C_TRUTH,
        lw=0.8,
        ls=":",
        drawstyle="steps-post",
        label="setpoint",
    )
    ax.plot(trace.t_min, trace.temp_out, color=C_AMB, lw=1.2, label="ambient sensor (model input)")
    ax.plot(trace.t_min, trace.temp, color=C_RAW, lw=0.6, label="chamber sensor, raw")
    ax.plot(trace.t_min, ma, color=C_MA, lw=1.4, label=f"moving average, {MA_WINDOW} readings")
    _band(ax, trace.t_min, res, allm)
    ax.plot(trace.t_min, res.estimate, color=C_KF, lw=1.4, label="Kalman estimate")
    ax.set_ylabel("temperature (°C)")
    ax.set_ylim(top=np.nanmax(trace.temp) + 2.5)  # headroom for the legend
    ax.set_title(
        "Real incubator day (3 Oct 2026, 1 367 readings at 15 s): raw, moving average, Kalman"
    )
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    s2 = 2 * np.sqrt(res.innovation_var)
    ax2.fill_between(trace.t_min, -s2, s2, color=C_KF, alpha=0.2, lw=0, label="±2√S expected")
    ax2.plot(
        trace.t_min,
        res.innovation,
        color=C_KF,
        lw=0.7,
        label="innovation: reading − model prediction",
    )
    ax2.set_ylabel("innovation (°C)")
    ax2.set_xlabel("minutes since first reading")
    ax2.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_zoom(
    trace: Trace, res: KalmanResult, ma: np.ndarray, model: ThermalModel, out: Path
) -> None:
    m = trace.window(*ZOOM)
    i0 = int(np.argmax(m))
    sub = Trace(trace.t_min[m], trace.temp[m], trace.temp_out[m], trace.heater[m], trace.target[m])
    # Open loop from the filter's estimate at the window start: the model with no sensor at all.
    # The heat input comes from the full relay history, so the lag state is right at the start.
    heat = model.heat_input(trace.heater, trace.t_min)
    open_loop = np.empty(m.sum())
    open_loop[0] = res.estimate[i0]
    for k in range(1, len(open_loop)):
        j = i0 + k
        open_loop[k] = model.step(
            open_loop[k - 1],
            trace.temp_out[j - 1],
            heat[j - 1],
            trace.t_min[j] - trace.t_min[j - 1],
        )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _heater_shading(ax, sub)
    ax.plot(
        sub.t_min,
        sub.temp,
        color=C_RAW,
        lw=1,
        drawstyle="steps-post",
        marker="o",
        ms=3,
        label="chamber sensor, raw (15 s steps, 0.01 °C counts)",
    )
    ax.plot(sub.t_min, ma[m], color=C_MA, lw=1.6, label=f"moving average, {MA_WINDOW} readings")
    ax.plot(sub.t_min, open_loop, color=C_TRUTH, lw=1, ls="--", label="model only (no sensor)")
    _band(ax, trace.t_min, res, m)
    ax.plot(sub.t_min, res.estimate[m], color=C_KF, lw=1.8, label="Kalman estimate")
    ax.set_xlabel("minutes since first reading")
    ax.set_ylabel("temperature (°C)")
    ax.set_title(
        "20 minutes at the 38.8 °C setpoint: heater pulses seen by sensor, model and filter"
    )
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def dropout_metrics(trace: Trace, kf: KalmanFilter1D, start: float, stop: float) -> dict:
    """Hide the chamber reading in [start, stop); score the filter against the hidden readings."""
    dropped = trace.with_dropout(start, stop)
    res = kf.run(dropped)
    held = hold_last(dropped.temp)
    m = trace.window(start, stop)
    hidden = trace.temp[m]
    err_kf, err_hold = res.estimate[m] - hidden, held[m] - hidden
    last = int(np.where(m)[0][-1])
    return {
        "window_min": [start, stop],
        "readings_hidden": int(m.sum()),
        "true_change_c": round(float(hidden[-1] - hidden[0]), 3),
        "kalman_rms_error_c": round(float(np.sqrt(np.mean(err_kf**2))), 3),
        "kalman_max_abs_error_c": round(float(np.max(np.abs(err_kf))), 3),
        "kalman_sigma_at_end_c": round(float(res.std[last]), 3),
        "hold_last_rms_error_c": round(float(np.sqrt(np.mean(err_hold**2))), 3),
        "hold_last_max_abs_error_c": round(float(np.max(np.abs(err_hold))), 3),
        "_res": res,
    }


def fig_dropout(trace: Trace, kf: KalmanFilter1D, metrics: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (name, met) in zip(axes, metrics.items(), strict=True):
        start, stop = met["window_min"]
        res = met["_res"]
        m = trace.window(start - 5, stop + 5)
        hid = trace.window(start, stop)
        held = hold_last(trace.with_dropout(start, stop).temp)
        _heater_shading(ax, trace, m)
        ax.axvspan(start, stop, color=C_RAW, alpha=0.08, lw=0, label="sensor dropout (10 min)")
        ax.plot(
            trace.t_min[m & ~hid],
            trace.temp[m & ~hid],
            "o",
            color=C_RAW,
            ms=3,
            label="reading seen",
        )
        ax.plot(
            trace.t_min[hid],
            trace.temp[hid],
            "o",
            mfc="none",
            color=C_RAW,
            ms=3,
            label="reading hidden (reference)",
        )
        ax.plot(trace.t_min[m], held[m], color=C_MA, lw=1.4, label="hold last value")
        _band(ax, trace.t_min, res, m)
        ax.plot(
            trace.t_min[m], res.estimate[m], color=C_KF, lw=1.8, label="Kalman (model only inside)"
        )
        ax.set_title(
            f"{name}: Kalman max error {met['kalman_max_abs_error_c']:.2f} °C, "
            f"hold-last {met['hold_last_max_abs_error_c']:.2f} °C",
            fontsize=9,
        )
        ax.set_xlabel("minutes since first reading")
    axes[0].set_ylabel("temperature (°C)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Simulated 10-minute chamber-sensor dropout on the real trace", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def synthetic_metrics(
    model: ThermalModel,
) -> tuple[dict, Trace, np.ndarray, KalmanResult, np.ndarray]:
    """Same recipe as on the real data: estimate r from the trace, tune q, then filter."""
    trace, truth = make_synthetic(model)
    r = estimate_measurement_std(trace.temp)
    q, _ = tune_process_noise(trace, model, r, Q_GRID)
    res, ma = KalmanFilter1D(model, q, r).run(trace), moving_average(trace.temp, MA_WINDOW)
    heating = model.heat_input(trace.heater, trace.t_min) > 0.5 * model.a

    def rmse(x: np.ndarray) -> float:
        return round(float(np.sqrt(np.mean((x - truth) ** 2))), 4)

    inside = np.abs(res.estimate - truth) <= 2 * res.std
    met = {
        "note": "synthetic.make_synthetic(seed 0): 240 min, setpoint 35 °C, sensor noise 0.03 °C",
        "measurement_noise_std_c": round(r, 4),
        "process_noise_std_c_per_sqrt_min": q,
        "rmse_c": {
            "raw": rmse(trace.temp),
            "moving_average": rmse(ma),
            "kalman": rmse(res.estimate),
        },
        "mean_error_while_heating_c": {
            "moving_average": round(float(np.mean((ma - truth)[heating])), 4),
            "kalman": round(float(np.mean((res.estimate - truth)[heating])), 4),
        },
        "truth_inside_2sigma_fraction": round(float(inside.mean()), 3),
    }
    return met, trace, truth, res, ma


def fig_synthetic(
    trace: Trace, truth: np.ndarray, res: KalmanResult, ma: np.ndarray, out: Path
) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    m = trace.window(20, 80)
    _heater_shading(ax, trace, m)
    ax.plot(
        trace.t_min[m],
        trace.temp[m],
        color=C_RAW,
        lw=0.8,
        drawstyle="steps-post",
        label="synthetic reading",
    )
    ax.plot(
        trace.t_min[m], ma[m], color=C_MA, lw=1.4, label=f"moving average, {MA_WINDOW} readings"
    )
    _band(ax, trace.t_min, res, m)
    ax.plot(trace.t_min[m], res.estimate[m], color=C_KF, lw=1.6, label="Kalman estimate")
    ax.plot(trace.t_min[m], truth[m], color=C_TRUTH, lw=1, ls="--", label="true temperature")
    ax.set_ylabel("temperature (°C)")
    ax.set_title("Synthetic trace with known truth (thermostat at 35 °C): 60 of 240 minutes")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax2.axhline(0, color=C_TRUTH, lw=0.6)
    ax2.plot(trace.t_min[m], (trace.temp - truth)[m], color=C_RAW, lw=0.6, label="raw − truth")
    ax2.plot(trace.t_min[m], (ma - truth)[m], color=C_MA, lw=1.2, label="moving average − truth")
    ax2.plot(trace.t_min[m], (res.estimate - truth)[m], color=C_KF, lw=1.4, label="Kalman − truth")
    ax2.set_ylabel("error vs truth (°C)")
    ax2.set_xlabel("minutes")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(fig_dir: Path = FIG_DIR) -> dict:
    _style()
    fig_dir.mkdir(exist_ok=True)
    trace = load_history()
    params = load_model_params()
    model = ThermalModel.from_params(params)
    r = estimate_measurement_std(trace.temp)
    q, scores = tune_process_noise(trace, model, r, Q_GRID)
    kf = KalmanFilter1D(model, q, r)
    res, ma = kf.run(trace), moving_average(trace.temp, MA_WINDOW)
    p_prior, p_post, k_ss = kf.steady_state(0.25)

    def roughness(x: np.ndarray) -> float:
        return round(float(np.sqrt(np.nanmean(np.diff(x) ** 2))), 4)

    drops = {name: dropout_metrics(trace, kf, *win) for name, win in DROPOUTS.items()}
    syn, s_trace, s_truth, s_res, s_ma = synthetic_metrics(model)

    fig_overview(trace, res, ma, fig_dir / "01_overview.png")
    fig_zoom(trace, res, ma, model, fig_dir / "02_zoom_heater_pulse.png")
    fig_dropout(trace, kf, drops, fig_dir / "03_dropout.png")
    fig_synthetic(s_trace, s_truth, s_res, s_ma, fig_dir / "04_synthetic_truth.png")

    results = {
        "source": "data/hive-1C5A00_history_2026-10-03.json and data/model_params.json, copied "
        "from Mariam-zeqred/incubator-telemetry-analysis; produced by `python -m kalman1d`",
        "readings": len(trace),
        "model": {
            "a_c_per_min": model.a,
            "b_per_min": model.b,
            "tau_h_min": model.tau_h,
            "tau_min": round(1 / model.b, 1),
        },
        "measurement_noise_std_c": round(r, 4),
        "process_noise_std_c_per_sqrt_min": q,
        "q_grid": {
            "min": float(Q_GRID[0]),
            "max": float(Q_GRID[-1]),
            "points": len(Q_GRID),
            "log_likelihood_at_best": round(float(scores.max()), 1),
        },
        "steady_state_15s": {
            "prior_std_c": round(np.sqrt(p_prior), 4),
            "posterior_std_c": round(np.sqrt(p_post), 4),
            "gain": round(k_ss, 4),
        },
        "nis_mean": round(res.nis(), 3),
        "roughness_rms_step_c": {
            "raw": roughness(trace.temp),
            "moving_average": roughness(ma),
            "kalman": roughness(res.estimate),
        },
        "dropout": {
            k: {kk: v for kk, v in d.items() if not kk.startswith("_")} for k, d in drops.items()
        },
        "synthetic": syn,
    }
    (fig_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
