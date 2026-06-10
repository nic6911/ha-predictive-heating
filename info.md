# Predictive Floor Heating (HeatPilot)

Adds a local, predictive control layer on top of your existing underfloor-heating
thermostats. Learns each room's thermal dynamics (grey-box RC model) and uses the
Home Assistant weather forecast plus an optional energy-price sensor to adjust
setpoints via Model Predictive Control — for more stable, cheaper heating.

- Fully local learning & prediction, no API keys
- Up to 16 zones
- Autonomous with guardrails (master switch, comfort bounds, manual-override back-off)
- Recognises when it has no control authority (sun/warm weather) and coasts
- Optional, toggleable price-aware scheduling

See the repository README for setup and rollout guidance.
