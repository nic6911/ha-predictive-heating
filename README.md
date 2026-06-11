# Predictive Floor Heating (HeatPilot)

A Home Assistant custom integration (HACS-compatible) that adds a **predictive
control layer** on top of your existing water/underfloor-heating thermostats
(e.g. Wavin Sentio via ESPHome). It learns each room's thermal behaviour and uses
**weather + (optional) energy-price forecasts** to adjust setpoints for a more
**stable and cheaper** heating performance — without you replacing anything.

It runs the learning and prediction **fully locally**. The only external input is
the weather forecast, which it reads from a Home Assistant weather entity (already
bound to your location). No cloud, no API keys.

## How it works

- **Grey-box RC thermal model** per zone — a physics-informed resistor/capacitor
  model of the room (cf. Bacher & Madsen 2011, *Identifying suitable models for the
  heat dynamics of buildings*). It is data-efficient and interpretable.
- **Online learning** with Recursive Least Squares, bootstrapped from your recorder
  history. The model keeps adapting over time.
- **Economic Model Predictive Control (MPC)** over a 12–24 h horizon. Each cycle it
  optimises the setpoint trajectory to keep the room inside your comfort band while
  exploiting free solar/outdoor heat and (optionally) shifting heating into cheaper /
  lower-CO₂ hours.
- **No-control-authority awareness**: when sun or mild weather already pushes the room
  above setpoint, it recognises it cannot cool the room and simply **coasts**, flagging
  this on a binary sensor instead of fighting physics.
- **Disturbance rejection**: sporadic events like an open window/door (the room cooling
  far faster than physics predicts) and sensor-fault spikes are detected and *excluded*
  from learning, so a transient never corrupts the model. While a disturbance is active
  the controller freezes learning and holds the last good setpoint until the room
  recovers (`binary_sensor.*_disturbance_detected`).

## Features

- Up to **16 zones/rooms**; unconfigured zones are ignored.
- Per zone you map: the **climate entity** (setpoint + temperature feedback), and
  optionally a separate temperature sensor, an outdoor-temperature sensor, and a solar
  irradiance sensor.
- **Autonomous with guardrails**: a master switch, per-zone comfort min/max, automatic
  back-off when you change a setpoint by hand, and an advisory mode that only
  *recommends* until the model has learned enough.
- Optional **price-aware** scheduling using any price sensor you already have
  (Nord Pool, Energi Data Service, Tibber, …).

## Entities

Per zone: `sensor.*_recommended_setpoint`, `sensor.*_predicted_temperature`,
`sensor.*_estimated_setpoint_reduction`, diagnostic `sensor.*_model_fit_rmse`,
diagnostic `sensor.*_prediction_error` (actual minus predicted),
`binary_sensor.*_manual_override`, `binary_sensor.*_coasting_on_free_heat`,
`binary_sensor.*_disturbance_detected`,
`switch.*_predictive_control`, `number.*_comfort_minimum/target/maximum`.

The `predicted_temperature` sensor also exposes a `forecast` attribute with the full
predicted indoor-temperature trajectory over the horizon (for graphing with ApexCharts).

Global: `switch` master *Predictive control*, `select` *Optimization profile*
(comfort / balanced / eco / price).

## Services

- `predictive_heating.train_now` — bootstrap/refresh the model from recorder history.
- `predictive_heating.reset_model` — discard a learned model.
- `predictive_heating.set_comfort_profile` — adjust a zone's comfort band at runtime.

## Installation (HACS)

1. HACS → Integrations → ⋮ → **Custom repositories** → add this repo URL, category
   *Integration*.
2. Install **Predictive Floor Heating (HeatPilot)** and restart Home Assistant.
3. Settings → Devices & Services → **Add Integration** → *Predictive Floor Heating*.
4. Pick your weather entity and global settings, then add zones from the integration's
   **Configure** (options) screen.

## Recommended rollout

1. Add your zones in **advisory** mode and watch the *Recommended setpoint* sensors for
   a day.
2. Run `predictive_heating.train_now` to bootstrap from history.
3. Once `model_fit_rmse` is below ~1 °C, flip zones (and the master switch) to autonomous.

## Development / testing

This repo follows the standard HA custom-integration workflow.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy homeassistant pytest-homeassistant-custom-component
pytest -q
```

To try it in a real instance without HACS, copy
`custom_components/predictive_heating` into your HA `config/custom_components/`
directory and restart Home Assistant. A `.devcontainer` is included for a one-command
dev instance (`Run Home Assistant` task), and CI runs `hassfest`, HACS validation and
the test suite.

## Disclaimer

This integration writes thermostat setpoints. Start in advisory mode, set sensible
comfort bounds, and keep the master switch handy. Use at your own risk.
