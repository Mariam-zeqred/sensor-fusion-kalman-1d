"""Synthetic trace with known truth: filter vs raw vs moving average, dropout, mismatch."""

import numpy as np

from kalman1d import KalmanFilter1D, ThermalModel, hold_last, make_synthetic, moving_average

MODEL = ThermalModel(a=0.59, b=0.0111, tau_h=4.9)
KF = KalmanFilter1D(MODEL, q=0.02, r=0.03)


def rmse(x: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((x - truth) ** 2)))


def test_generator_is_deterministic_and_the_thermostat_holds_the_setpoint():
    a, truth_a = make_synthetic(MODEL, seed=0)
    b, truth_b = make_synthetic(MODEL, seed=0)
    assert np.array_equal(a.temp, b.temp) and np.array_equal(truth_a, truth_b)
    settled = a.t_min > 60
    assert abs(truth_a[settled].mean() - 35.0) < 0.5
    assert 0.05 < a.heater[settled].mean() < 0.5
    # readings are quantised to 0.01 °C
    assert np.allclose(a.temp * 100, np.round(a.temp * 100))


def test_filter_beats_raw_and_moving_average_against_truth():
    trace, truth = make_synthetic(MODEL, seed=0)
    est = KF.run(trace).estimate
    e_raw, e_ma, e_kf = (
        rmse(trace.temp, truth),
        rmse(moving_average(trace.temp, 8), truth),
        rmse(est, truth),
    )
    assert e_kf < 0.7 * e_raw
    assert e_kf < 0.25 * e_ma


def test_two_sigma_band_covers_the_truth_about_95_percent_of_the_time():
    trace, truth = make_synthetic(MODEL, seed=0)
    res = KF.run(trace)
    inside = np.abs(res.estimate - truth) <= 2 * res.std
    assert 0.90 < inside.mean() < 0.99


def test_moving_average_lags_during_heating_and_the_filter_does_not():
    trace, truth = make_synthetic(MODEL, seed=0)
    heating = MODEL.heat_input(trace.heater, trace.t_min) > 0.5 * MODEL.a
    bias_ma = np.mean((moving_average(trace.temp, 8) - truth)[heating])
    bias_kf = np.mean((KF.run(trace).estimate - truth)[heating])
    assert bias_ma < -0.2  # the 2-min average trails a 0.5 °C/min ramp by about a minute
    assert abs(bias_kf) < 0.02


def test_dropout_during_heating_is_bridged_by_the_model():
    trace, truth = make_synthetic(MODEL, seed=0)
    heating = np.where(MODEL.heat_input(trace.heater, trace.t_min) > 0.5 * MODEL.a)[0]
    start = trace.t_min[heating[heating > 240][0]]  # a heating phase after the warm-up
    dropped = trace.with_dropout(start, start + 5)
    gap = trace.window(start, start + 5)
    res = KF.run(dropped)
    err_kf = np.abs(res.estimate - truth)[gap]
    err_hold = np.abs(hold_last(dropped.temp) - truth)[gap]
    assert truth[gap][-1] - truth[gap][0] > 0.5  # the chamber really moved meanwhile
    assert err_hold.max() > 0.5
    assert err_kf.max() < 0.25
    assert res.std[gap][-1] > 2 * res.std[gap][0]  # uncertainty grows while coasting


def test_mismatched_model_parameters_still_beat_the_raw_reading():
    trace, truth = make_synthetic(MODEL, seed=8)
    wrong = ThermalModel(a=MODEL.a * 1.15, b=MODEL.b * 0.85, tau_h=MODEL.tau_h * 1.3)
    est = KalmanFilter1D(wrong, q=0.03, r=0.03).run(trace).estimate
    assert rmse(est, truth) < 0.85 * rmse(trace.temp, truth)
