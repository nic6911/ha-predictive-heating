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
CONF_PRICE_OPTIMIZE = "price_optimize"
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

# Disturbance / outlier rejection (windows, doors, sporadic transients)
# One-step residual rejection thresholds for the online RLS estimator.
OUTLIER_SIGMA = 4.0  # reject if |residual| > OUTLIER_SIGMA * robust scale
OUTLIER_ABS_CAP = 1.5  # deg C; also reject if |residual| exceeds this absolute cap
# A window/door disturbance is flagged when the measured temperature change falls
# far *below* what the model expected (room cooling abnormally fast).
DISTURBANCE_DROP_SIGMA = 4.0
DISTURBANCE_DROP_MIN = 0.5  # deg C; minimum drop-below-prediction to flag
# Once a disturbance is flagged we freeze learning and hold the last good setpoint
# for at least this long, or until the temperature recovers.
DISTURBANCE_HOLD = timedelta(minutes=60)

# Storage
STORAGE_VERSION = 1
STORAGE_KEY = "predictive_heating_models"

# Dispatcher signal
SIGNAL_UPDATE = "predictive_heating_update"
