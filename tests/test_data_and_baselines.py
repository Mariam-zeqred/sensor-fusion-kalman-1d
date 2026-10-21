"""The real file loads as documented; the thermal model and baselines behave."""

import numpy as np
import pytest

from kalman1d import ThermalModel, hold_last, load_history, load_model_params, moving_average


def test_real_history_loads_as_documented():
    trace = load_history()
    assert len(trace) == 1367
    assert np.all(np.diff(trace.t_min) > 0)
    assert np.median(trace.dt_min) == pytest.approx(0.25)
    assert trace.dt_min.max() == pytest.approx(74 / 60)
    assert not np.isnan(trace.temp).any() and not np.isnan(trace.temp_out).any()
    assert set(np.unique(trace.heater)) == {0.0, 1.0}
    assert 26 < trace.temp.min() < trace.temp.max() < 39.5
    assert set(np.unique(trace.target)) == {30.0, 32.0, 34.5, 38.8}


def test_model_params_are_the_source_repo_heater_lag_fit():
    model = ThermalModel.from_params(load_model_params())
    assert model.a == pytest.approx(0.591, abs=0.005)
    assert model.b == pytest.approx(0.0111, abs=0.0005)
    assert model.tau_h == pytest.approx(4.9, abs=0.1)


def test_with_dropout_masks_exactly_the_window():
    trace = load_history()
    dropped = trace.with_dropout(190, 200)
    gap = trace.window(190, 200)
    assert np.isnan(dropped.temp[gap]).all()
    assert np.array_equal(dropped.temp[~gap], trace.temp[~gap])
    assert not np.isnan(trace.temp).any()  # the original is untouched


def test_heat_input_is_a_first_order_lag_of_the_relay():
    model = ThermalModel(a=0.6, b=0.01, tau_h=5.0)
    t = np.arange(0, 60, 0.25)
    heat = model.heat_input(np.ones_like(t), t)
    assert heat[0] == 0.0
    k = int(np.argmin(np.abs(t - 5.0)))
    assert heat[k] == pytest.approx(0.6 * (1 - np.exp(-1)), rel=1e-6)
    assert heat[-1] == pytest.approx(0.6, abs=1e-4)
    # the exact update does not care how the interval is split
    assert model.heat_step(0.0, 1.0, 1.0) == pytest.approx(
        model.heat_step(model.heat_step(0.0, 1.0, 0.5), 1.0, 0.5)
    )


def test_open_loop_model_settles_where_heating_balances_losses():
    model = ThermalModel(a=0.6, b=0.01, tau_h=5.0)
    t = np.arange(0, 1500, 0.25)
    temp = model.simulate(25.0, np.full_like(t, 25.0), np.ones_like(t), t)
    assert temp[-1] == pytest.approx(25.0 + 0.6 / 0.01, rel=1e-3)


def test_moving_average_and_hold_last():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.allclose(moving_average(x, 2), [1.0, 1.5, 2.0, 4.0, 4.5])
    assert np.allclose(hold_last(x), [1.0, 2.0, 2.0, 4.0, 5.0])
    assert np.isnan(moving_average(np.array([np.nan, np.nan]), 2)).all()
