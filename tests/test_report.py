"""Smoke test: the full report runs on the real data and its headline claims hold."""

import json

from kalman1d.report import main


def test_report_writes_figures_and_consistent_results(tmp_path):
    results = main(tmp_path)
    for name in ("01_overview", "02_zoom_heater_pulse", "03_dropout", "04_synthetic_truth"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000
    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["readings"] == 1367 == results["readings"]
    assert 0.005 < results["measurement_noise_std_c"] < 0.05
    assert 0.5 < results["nis_mean"] < 1.5
    ramp = results["dropout"]["ramp"]
    assert ramp["kalman_max_abs_error_c"] < 0.2 * ramp["hold_last_max_abs_error_c"]
    assert results["synthetic"]["rmse_c"]["kalman"] < results["synthetic"]["rmse_c"]["raw"]
    assert 0.9 < results["synthetic"]["truth_inside_2sigma_fraction"] < 0.99
