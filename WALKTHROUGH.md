# Walkthrough (for Mariam)

## Why it is built this way

The incubator has three signals about the chamber temperature: the chamber SHT31 (the
measurement), the ambient SHT31 (sets how fast the chamber cools) and the heater relay (sets
how fast it warms). A Kalman filter combines a physical model with a noisy measurement, so
this repo uses the thermal model fitted in incubator-telemetry-analysis as the *prediction*
and the chamber sensor as the *correction*. The state is one number, `T`. The
ambient reading and the relay are inputs to the prediction, not measurements of `T`; that is
the trick that keeps the filter one-dimensional while still fusing three signals.

The first version used the first-order model with a 3.5-minute dead time. It over-predicted
every short heater pulse by half a degree, the tuner compensated with a huge process noise,
and the filter degenerated into "trust the sensor". The heater-lag model (heat reaches the
chamber through a 4.9-minute lag) fixed that; the lag state `E` depends only on the relay
history, so it is computed up front and the Kalman state stays scalar.

The honest headline: the SHT31 is quiet (0.012 °C), so on a healthy sensor the filter smooths
little. Its value is the uncertainty it reports and the dropout case: it coasted through ten
minutes of a 3 °C heating ramp with 0.07 °C error.

## File map

- `src/kalman1d/data.py`: loads the JSON into a `Trace` (minutes, chamber, ambient, relay,
  setpoint); `with_dropout()` blanks the chamber reading for an interval.
- `src/kalman1d/model.py`: `ThermalModel` with `heat_input()` (lagged heat from the relay),
  `step()` (one Euler step of `T`) and `simulate()` (open loop).
- `src/kalman1d/filter.py`: `KalmanFilter1D.run()`, the predict/update loop;
  `steady_state()`, the closed-form Riccati answer; `estimate_measurement_std()` and
  `tune_process_noise()`, which get `r` and `q` from the data.
- `src/kalman1d/baselines.py`: moving average and hold-last.
- `src/kalman1d/synthetic.py`: thermostat simulation with a known truth, for the tests.
- `src/kalman1d/report.py`: runs everything on the real day, writes the four figures and
  `figures/results.json`, the source of every README number.
- `tests/`: filter algebra (Riccati fixed point, the `r → ∞` and `q → ∞` limits, missing
  readings), synthetic truth (accuracy, coverage, lag, dropout, wrong parameters), data
  loading, report smoke run.

## Three questions a professor may ask

**Why is the measurement noise so small, and how do you know it?** It is estimated from the
data: the second difference of a smooth signal plus white noise has variance `6r²`. The
median-based estimate gives 0.012 °C, about one count of the sensor's 0.01 °C resolution.
With `r` this small the gain is 0.51 at 15 s: sensor and model are believed about equally.

**How did you choose the process noise, and how do you know the filter is consistent?** `q`
maximises the likelihood of the innovations (reading minus prediction) over a grid. Two checks:
the mean normalised innovation squared is 0.92 (should be near 1), and on the synthetic trace
the truth lies inside the ±2σ band 95.8 % of the time. The same recipe recovers the
generator's `r` and `q` there.

**What breaks it?** A stuck or offset sensor: the filter believes it. The fix is innovation
gating (ignore readings more than 3√S from the prediction), which the dropout path makes easy.
Also the model parameters are in-sample, fitted on this same day; another day will need a
larger `q`.

## One thing to extend

Make the state two-dimensional, `(T, E)`, so the heater's effective power is estimated rather
than assumed; `E` can then absorb heater ageing or a wrong relay reading. The maths is a 2×2
version of the same predict/update loop, and the synthetic generator can be given a drifting
`a` to test it.
