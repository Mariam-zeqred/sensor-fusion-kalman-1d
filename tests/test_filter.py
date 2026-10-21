"""Filter algebra on traces where the answer is known."""

import numpy as np
import pytest

from kalman1d import KalmanFilter1D, ThermalModel, Trace, estimate_measurement_std, make_synthetic
from kalman1d.filter import tune_process_noise

MODEL = ThermalModel(a=0.59, b=0.0111, tau_h=4.9)


def constant_trace(n: int = 400, temp: float = 30.0, noise: float = 0.02, seed: int = 1) -> Trace:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.25
    return Trace(
        t, temp + noise * rng.normal(size=n), np.full(n, temp), np.zeros(n), np.full(n, temp)
    )


def test_variance_and_gain_settle_to_the_riccati_fixed_point():
    kf = KalmanFilter1D(MODEL, q=0.02, r=0.02)
    res = kf.run(constant_trace())
    p_prior, p_post, k_ss = kf.steady_state(0.25)
    assert res.variance[-1] == pytest.approx(p_post, rel=1e-6)
    assert res.gain[-1] == pytest.approx(k_ss, rel=1e-6)
    assert res.innovation_var[-1] == pytest.approx(p_prior + 0.02**2, rel=1e-6)
    assert 0.0 < k_ss < 1.0


def test_huge_measurement_noise_means_the_estimate_is_the_open_loop_model():
    trace, _ = make_synthetic(MODEL, duration_min=60, seed=2)
    kf = KalmanFilter1D(MODEL, q=0.001, r=1e6)
    res = kf.run(trace, x0=trace.temp[0], p0=0.0)
    open_loop = MODEL.simulate(trace.temp[0], trace.temp_out, trace.heater, trace.t_min)
    assert np.max(np.abs(res.estimate - open_loop)) < 1e-3


def test_huge_process_noise_means_the_estimate_is_the_reading():
    trace, _ = make_synthetic(MODEL, duration_min=60, seed=3)
    res = KalmanFilter1D(MODEL, q=1e3, r=0.03).run(trace)
    assert np.max(np.abs(res.estimate - trace.temp)) < 1e-4
    assert np.all(res.gain[1:] > 0.999)  # step 0 starts from p0 = r², hence gain 0.5


def test_missing_readings_coast_on_the_model_and_grow_the_variance():
    trace, _ = make_synthetic(MODEL, duration_min=90, seed=4)
    gap = trace.window(40, 50)
    dropped = trace.with_dropout(40, 50)
    res = KalmanFilter1D(MODEL, q=0.02, r=0.03).run(dropped)
    idx = np.where(gap)[0]
    assert np.all(res.gain[idx] == 0)
    assert np.all(np.isnan(res.innovation[idx]))
    assert np.array_equal(res.estimate[idx], res.predicted[idx])
    assert np.all(np.diff(res.variance[idx]) > 0)
    # the variance comes back down once readings return
    assert res.variance[idx[-1] + 5] < res.variance[idx[-1]] / 3


def test_a_longer_step_adds_more_process_variance():
    t = np.array([0.0, 0.25, 0.5, 1.75, 2.0])  # one 75 s gap, as in the real file
    trace = Trace(t, np.full(5, np.nan), np.full(5, 28.0), np.zeros(5), np.full(5, 28.0))
    res = KalmanFilter1D(MODEL, q=0.02, r=0.02).run(trace, x0=30.0, p0=0.0)
    growth = np.diff(res.variance)
    assert growth[2] > 4 * growth[1]


def test_measurement_noise_estimate_recovers_white_noise_on_a_slow_ramp():
    rng = np.random.default_rng(5)
    signal = 30 + 0.3 * np.arange(2000) / 4  # 0.3 °C/min at 15 s steps
    noise = 0.05
    r_hat = estimate_measurement_std(signal + noise * rng.normal(size=2000))
    assert r_hat == pytest.approx(noise, rel=0.15)


def test_process_noise_tuning_recovers_the_generator_value():
    trace, _ = make_synthetic(MODEL, process_std=0.01, noise_std=0.03, seed=6)
    q_true = 0.01 / np.sqrt(0.25)  # per-step std -> per sqrt(minute)
    r_hat = estimate_measurement_std(trace.temp)
    q_hat, scores = tune_process_noise(trace, MODEL, r_hat, np.geomspace(0.002, 0.2, 25))
    assert q_true / 1.6 < q_hat < q_true * 1.6
    assert scores.argmax() not in (0, len(scores) - 1)


def test_nis_is_close_to_one_when_the_noise_levels_are_right():
    trace, _ = make_synthetic(MODEL, process_std=0.01, noise_std=0.03, seed=7)
    res = KalmanFilter1D(MODEL, q=0.02, r=0.03).run(trace)
    assert 0.8 < res.nis() < 1.25
