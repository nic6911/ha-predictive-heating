"""Constants for the Predictive Floor Heating integration."""

from __future__ import annotations

from datetime import timedelta

DOMAIN = "predictive_heating"
PLATFORMS = ["sensor", "binary_sensor", "switch", "number", "select", "button"]

# Limits
MAX_ZONES = 16

# Config keys -- global
CONF_WEATHER_ENTITY = "weather_entity"
CONF_UPDATE_INTERVAL = "update_interval_minutes"
CONF_HORIZON_HOURS = "horizon_hours"
CONF_STEP_MINUTES = "step_minutes"
CONF_MODE = "mode"
CONF_PRICE_ENTITY = "price_entity"
CONF_CO2_ENTITY = "co2_entity"
CONF_CO2_OPTIMIZE = "co2_optimize"
CONF_PRICE_OPTIMIZE = "price_optimize"
CONF_MODEL_TYPE = "model_type"
CONF_ZONES = "zones"

# Config keys -- per zone
CONF_ZONE_ID = "zone_id"
CONF_CLIMATE_ENTITY = "climate_entity"
CONF_TEMP_SENSOR = "temp_sensor"
CONF_OUTDOOR_SENSOR = "outdoor_sensor"
CONF_IRRADIANCE_SENSOR = "irradiance_sensor"
CONF_COMFORT_MIN = "comfort_min"
CONF_COMFORT_MAX = "comfort_max"
CONF_COMFORT_TARGET = "comfort_target"
CONF_ZONE_MODE = "zone_mode"
CONF_ZONE_ENABLED = "enabled"

# Modes (global optimization profile)
MODE_COMFORT = "comfort"
MODE_BALANCED = "balanced"
MODE_ECO = "eco"
MODE_PRICE = "price"
MODES = [MODE_COMFORT, MODE_BALANCED, MODE_ECO, MODE_PRICE]

# Per-zone control mode
ZONE_MODE_AUTO = "auto"
ZONE_MODE_ADVISORY = "advisory"
ZONE_MODES = [ZONE_MODE_AUTO, ZONE_MODE_ADVISORY]

# Defaults
DEFAULT_UPDATE_INTERVAL = 15  # minutes
DEFAULT_HORIZON_HOURS = 24
DEFAULT_STEP_MINUTES = 30
DEFAULT_MODE = MODE_BALANCED
DEFAULT_COMFORT_MIN = 19.0
DEFAULT_COMFORT_MAX = 23.0
DEFAULT_COMFORT_TARGET = 21.0
DEFAULT_PRICE_OPTIMIZE = False
DEFAULT_CO2_OPTIMIZE = False
DEFAULT_MODEL_TYPE = "standard"

# Model types
MODEL_STANDARD = "standard"
MODEL_ENHANCED = "enhanced"
MODEL_3R2C = "3r2c"
MODEL_TYPES = [MODEL_STANDARD, MODEL_ENHANCED, MODEL_3R2C]

# Plausible indoor/setpoint temperature band (deg C). Readings outside this range
# are treated as sensor faults (e.g. the 327.67 C sentinel some ESPHome/Wavin
# sensors emit) and ignored so they never poison learning or control.
PLAUSIBLE_TEMP_MIN = -10.0
PLAUSIBLE_TEMP_MAX = 50.0

# Control behaviour
SETPOINT_DEADBAND = 0.2  # deg C; only write if change exceeds this
MANUAL_OVERRIDE_TOLERANCE = 0.05  # deg C tolerance for detecting external setpoint change
MANUAL_OVERRIDE_HOLD = timedelta(hours=2)  # back off autonomous control after manual change

# Objective weights per mode: (w_comfort, w_energy)
MODE_WEIGHTS = {
    MODE_COMFORT: (10.0, 0.1),
    MODE_BALANCED: (3.0, 1.0),
    MODE_ECO: (1.0, 3.0),
    MODE_PRICE: (1.0, 6.0),
}

# Model fit quality: RMSE (deg C) below which autonomous control is allowed
FIT_RMSE_AUTONOMY_THRESHOLD = 1.0

# Rolling buffer + periodic batch re-identification.
# We forecast and control from a STABLE, periodically re-fitted robust batch model
# rather than from the fast online RLS estimator. The online RLS minimises one-step
# error and -- being weakly regularised -- drifts into parameter combinations that
# are fine one-step but explode in the multi-step open-loop rollout the MPC depends
# on (validated on a year of data: 24 h open-loop RMSE ~17 C for online RLS vs
# ~1 C for periodic batch refit + offset-free bias). The buffer keeps the most
# recent few weeks of accepted transitions; we re-fit on a slow cadence. These are
# generic robustness/compute choices, not building-specific tuning -- the buffer
# length only needs to span enough seasons-worth of excitation, and validation
# showed accuracy is insensitive to it (10..60 days all work).
BUFFER_DAYS = 45
REFIT_INTERVAL_HOURS = 6
# Need at least this many accepted transitions before the first batch refit.
MIN_REFIT_SAMPLES = 48

# CO2 optimisation: scale factor converting CO2 intensity to an equivalent
# price multiplier. At CO2_SCALE=1.0 and BASELINE=200 g/kWh, a reading of
# 200 g/kWh leaves the price unchanged; 400 g/kWh doubles the effective cost.
CO2_SCALE = 0.5
CO2_BASELINE = 200.0  # g/kWh

# Disturbance / outlier rejection (windows, doors, sporadic transients)
#
# Graduated innovation gating replaces the old binary accept/reject for the
# online RLS estimator. Three zones:
#   |residual| <= GRADUATED_THRESHOLD_LOW  -> full update
#   GRADUATED_THRESHOLD_LOW < |residual| <= GRADUATED_THRESHOLD_HIGH
#                                         -> discounted update (linear taper)
#   |residual| > GRADUATED_THRESHOLD_HIGH -> rejected UNLESS sustained
# A sustained error (GRADUATED_SUSTAINED_COUNT of last GRADUATED_WINDOW
# samples exceed GRADUATED_THRESHOLD_HIGH) indicates a genuine regime change
# (e.g. a seasonal outdoor shift) and is accepted (heavily discounted).
GRADUATED_THRESHOLD_LOW = 0.5  # deg C; above this, discount the RLS update
GRADUATED_THRESHOLD_HIGH = 1.5  # deg C; above this, require sustained evidence
GRADUATED_SUSTAINED_COUNT = 3  # how many samples in window must exceed HIGH
GRADUATED_WINDOW = 5  # rolling window length for sustained check
# Legacy binary-gate thresholds (kept for disturbance detection and reference).
OUTLIER_SIGMA = 4.0  # reject if |residual| > OUTLIER_SIGMA * robust scale
OUTLIER_ABS_CAP = 1.5  # deg C; also reject if |residual| exceeds this absolute cap
# A window/door disturbance is flagged when the measured temperature change falls
# far *below* what the model expected (room cooling abnormally fast).
DISTURBANCE_DROP_SIGMA = 4.0
DISTURBANCE_DROP_MIN = 0.5  # deg C; minimum drop-below-prediction to flag
# Once a disturbance is flagged we freeze learning and hold the last good setpoint
# for at least this long, or until the temperature recovers.
DISTURBANCE_HOLD = timedelta(minutes=60)

# Thermal-inertia (enhanced) model -- adds one parameter k_mem that captures
# the slab/floor thermal-mass effect: if the room was warming last step it
# tends to continue warming. Bounded to [0, 0.9] for stability.
PARAM_MEM_LOWER = 0.0
PARAM_MEM_UPPER = 0.9
DEFAULT_K_MEM = 0.15

# Number of parameters in each model type.
N_PARAMS_STANDARD = 4  # [ka, ks, kh, kg]
N_PARAMS_ENHANCED = 5  # [ka, ks, kh, kg, k_mem]
N_PARAMS_3R2C = 6  # [ka, ks, kh, kg, k_aw, k_wa] for 3R2C two-node model

# Hourly bias (time-varying internal gains schedule).
# Number of hours in a day for the per-hour bias array.
N_HOURS = 24
# EWMA learning rate for online per-hour bias updates. Each hour bucket gets
# ~2 updates/day at 30-min steps; alpha=0.1 gives ~10-update (~5 day) time constant.
HBIAS_ALPHA = 0.1

# Storage
STORAGE_VERSION = 1
STORAGE_KEY = "predictive_heating_models"

# Dispatcher signal
SIGNAL_UPDATE = "predictive_heating_update"
