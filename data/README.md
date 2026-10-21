# Data

Both files are copied unchanged from
[incubator-telemetry-analysis](https://github.com/Mariam-zeqred/incubator-telemetry-analysis)
(commit `eef528e`). They come from one real queen-bee incubator (device `hive-1C5A00`).

| File | What it is |
|---|---|
| `hive-1C5A00_history_2026-10-03.json` | 1 367 readings at 15 s on 3 Oct 2026 (10:02–15:45 UTC). Fields used here: `ts` (epoch s, UTC), `temp` (chamber SHT31, °C), `tempOut` (ambient SHT31, °C), `heater` (relay on/off), `targetTemp` (setpoint). The setpoint was stepped 32 → 34.5 → 38.8 → 30 °C during the day. |
| `model_params.json` | Thermal-model parameters fitted in the source repo on this same trace. This repo uses the `second_order` block (heater lag): `a` = 0.591 °C/min, `b` = 0.0111 1/min, `τh` = 4.9 min. |

Nothing in this folder is synthetic. The synthetic trace used by the tests is generated
by `src/kalman1d/synthetic.py` at test time and is not stored.
