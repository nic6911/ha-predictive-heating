#!/usr/bin/env python3
"""HeatPilot performance report from Home Assistant recorder history.

Read-only. Uses the Home Assistant REST API and the Python standard library.

Examples
--------
    python tools/heatpilot_report.py
    python tools/heatpilot_report.py --start 2026-10-09T19:00:00Z \
        --end 2026-11-04T00:00:00Z --start-b 2026-11-04T00:00:00Z
    python tools/heatpilot_report.py --zones bad_born,stue_ka --json-out baseline.json

The token is read from ``--token-file``, ``HA_TOKEN_PATH`` or
``%LOCALAPPDATA%\\ha_token.txt``. It is never printed.

Metrics per window and zone: mean/min/max room temperature, time below the
comfort minimum and above the comfort maximum, mean distance from target,
time coasting, time in (manual) override, and model RMSE. When a district
heating daily-energy sensor and an outdoor sensor are available, the script
also reports kWh per heating degree day (base 15.5 C).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from statistics import mean, median

DEFAULT_BASE_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
DEFAULT_ENERGY = "sensor.eforsyning_energy_used"
DEFAULT_OUTDOOR = "sensor.hue_outdoor_motion_sensor_1_temperature"
SAMPLE_MINUTES = 10
HDD_BASE = 15.5


# --------------------------------------------------------------------------- api
class Api:
    def __init__(self, base_url: str, token: str, timeout: float = 60.0) -> None:
        self.base = base_url.rstrip("/") + "/api"
        self.token = token
        self.timeout = timeout

    def get(self, path: str, **params) -> object:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", "replace")[:300]
            raise SystemExit(f"HA API {err.code} for {path}: {body}") from err
        except urllib.error.URLError as err:
            raise SystemExit(f"Cannot reach Home Assistant at {self.base}: {err}") from err

    def states(self) -> list[dict]:
        return self.get("/states")  # type: ignore[return-value]

    def history(self, entity_id: str, start: datetime, end: datetime) -> list[dict]:
        params = {
            "filter_entity_id": entity_id,
            "end_time": _iso(end),
            "minimal_response": "",
            "no_attributes": "",
        }
        payload = self.get(f"/history/period/{_iso(start)}", **params)
        if not isinstance(payload, list) or not payload:
            return []
        return payload[0] if isinstance(payload[0], list) else []


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _parse_time(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _row_time(row: dict) -> datetime:
    raw = row.get("last_changed") or row.get("last_updated")
    if not raw:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return _parse_time(str(raw))


def _sample(rows: list[dict], start: datetime, end: datetime, minutes: int) -> list[object]:
    """Zero-order-hold the entity onto a fixed grid; values may be None."""
    out: list[object] = []
    index = 0
    current = None
    when = start
    while when <= end:
        while index < len(rows) and _row_time(rows[index]) <= when:
            current = rows[index].get("state")
            index += 1
        out.append(current)
        when += timedelta(minutes=minutes)
    return out


def _numbers(values: list[object]) -> list[float]:
    return [n for n in (_float(v) for v in values) if n is not None]


def _temperatures(values: list[object]) -> list[float]:
    """Like _numbers, but drops sensor-fault sentinels (0 C, 327.67 C)."""
    return [n for n in _numbers(values) if 0.0 < n < 50.0]


def _on_share(values: list[object]) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(1 for v in values if str(v).lower() == "on") / len(values)


# ------------------------------------------------------------------------ zones
def discover_zones(states: list[dict]) -> dict[str, dict]:
    """Map zone_id -> entity ids from the integration's entities."""
    zones: dict[str, dict] = {}
    for state in states:
        entity_id = str(state.get("entity_id", ""))
        if not entity_id.startswith("sensor.heatpilot_"):
            continue
        if not entity_id.endswith("_recommended_setpoint"):
            continue
        zone_id = entity_id[len("sensor.heatpilot_") : -len("_recommended_setpoint")]
        zones[zone_id] = {
            "recommended": entity_id,
            "temperature": f"sensor.{zone_id}_temperatur",
            "coasting": f"binary_sensor.heatpilot_{zone_id}_coasting_on_free_heat",
            "override": f"binary_sensor.heatpilot_{zone_id}_manual_override_active",
            "applied": f"binary_sensor.heatpilot_{zone_id}_setpoint_applied",
            "rmse": f"sensor.heatpilot_{zone_id}_model_fit_rmse",
            "prediction_error": (
                f"sensor.heatpilot_{zone_id}_prediction_error_actual_minus_predicted"
            ),
        }
    return zones


def _state_value(states: list[dict], entity_id: str) -> str | None:
    for state in states:
        if state.get("entity_id") == entity_id:
            return str(state.get("state"))
    return None


# ---------------------------------------------------------------------- metrics
def zone_metrics(
    api: Api,
    zone_id: str,
    zone: dict,
    start: datetime,
    end: datetime,
    comfort_min: float,
    comfort_max: float,
    comfort_target: float,
) -> dict:
    temps = _temperatures(
        _sample(api.history(zone["temperature"], start, end), start, end, SAMPLE_MINUTES)
    )
    if not temps:
        return {"zone": zone_id, "samples": 0}

    coast = _on_share(_sample(api.history(zone["coasting"], start, end), start, end, SAMPLE_MINUTES))
    override = _on_share(_sample(api.history(zone["override"], start, end), start, end, SAMPLE_MINUTES))
    applied = _on_share(_sample(api.history(zone["applied"], start, end), start, end, SAMPLE_MINUTES))

    rmse_rows = api.history(zone["rmse"], start, end)
    rmse_values = _numbers([row.get("state") for row in rmse_rows])
    error_rows = api.history(zone["prediction_error"], start, end)
    errors = _numbers([row.get("state") for row in error_rows])
    recommended = _numbers(
        _sample(api.history(zone["recommended"], start, end), start, end, SAMPLE_MINUTES)
    )

    return {
        "zone": zone_id,
        "samples": len(temps),
        "mean": mean(temps),
        "min": min(temps),
        "max": max(temps),
        "below_min_pct": 100.0 * sum(1 for t in temps if t < comfort_min - 0.05) / len(temps),
        "above_max_pct": 100.0 * sum(1 for t in temps if t > comfort_max + 0.05) / len(temps),
        "mean_abs_dev": mean(abs(t - comfort_target) for t in temps),
        "mean_recommended": mean(recommended) if recommended else None,
        "coasting_pct": coast,
        "override_pct": override,
        "applied_pct": applied,
        "rmse_last": rmse_values[-1] if rmse_values else None,
        "rmse_mean": mean(rmse_values) if rmse_values else None,
        "prediction_error_mean": mean(errors) if errors else None,
        "prediction_error_abs_max": max((abs(e) for e in errors), default=None),
    }


def daily_energy(api: Api, entity_id: str, start: datetime, end: datetime) -> dict[str, float]:
    """Daily maximum of a sensor that resets each day."""
    rows = api.history(entity_id, start, end)
    daily: dict[str, float] = {}
    for row in rows:
        value = _float(row.get("state"))
        if value is None:
            continue
        day = _row_time(row).date().isoformat()
        daily[day] = max(daily.get(day, value), value)
    return daily


def daily_mean(api: Api, entity_id: str, start: datetime, end: datetime) -> dict[str, float]:
    rows = api.history(entity_id, start, end)
    acc: dict[str, list[float]] = {}
    for row in rows:
        value = _float(row.get("state"))
        if value is None:
            continue
        acc.setdefault(_row_time(row).date().isoformat(), []).append(value)
    return {day: mean(values) for day, values in acc.items()}


def energy_summary(api: Api, energy_entity: str, outdoor_entity: str, start: datetime, end: datetime) -> dict:
    energy = daily_energy(api, energy_entity, start, end)
    outdoor = daily_mean(api, outdoor_entity, start, end)
    days = sorted(set(energy) & set(outdoor))
    if not days:
        return {"days": 0}
    kwh = [energy[d] for d in days]
    hdd = [max(0.0, HDD_BASE - outdoor[d]) for d in days]
    total_hdd = sum(hdd)
    total_kwh = sum(kwh)
    return {
        "days": len(days),
        "kwh_total": total_kwh,
        "kwh_per_day": total_kwh / len(days),
        "kwh_median": median(kwh),
        "hdd_total": total_hdd,
        "kwh_per_hdd": (total_kwh / total_hdd) if total_hdd else None,
        "outdoor_mean": mean(outdoor[d] for d in days),
    }


# ----------------------------------------------------------------------- output
def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n### {title}\n")
    header = (
        "| Zone | samples | mean | min | max | <min % | >max % | |dev| | rec.sp | "
        "coast % | override % | applied % | RMSE | bias |"
    )
    print(header)
    print("|" + "---|" * 14)
    for row in rows:
        if not row.get("samples"):
            print(
                f"| {row['zone']} | 0 | - | - | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        print(
            "| {zone} | {n} | {mean} | {mn} | {mx} | {below} | {above} | {dev} | {rec} | "
            "{coast} | {override} | {applied} | {rmse} | {bias} |".format(
                zone=row["zone"],
                n=row["samples"],
                mean=_fmt(row["mean"]),
                mn=_fmt(row["min"]),
                mx=_fmt(row["max"]),
                below=_fmt(row["below_min_pct"], 1),
                above=_fmt(row["above_max_pct"], 1),
                dev=_fmt(row["mean_abs_dev"]),
                rec=_fmt(row["mean_recommended"]),
                coast=_fmt(row["coasting_pct"], 0),
                override=_fmt(row["override_pct"], 0),
                applied=_fmt(row["applied_pct"], 0),
                rmse=_fmt(row["rmse_last"], 3),
                bias=_fmt(row["prediction_error_mean"]),
            )
        )


def print_energy(title: str, summary: dict) -> None:
    print(f"\n### {title}\n")
    if not summary.get("days"):
        print("No overlapping energy/outdoor days.")
        return
    print(
        "| days | kWh | kWh/day | median kWh | HDD (base {base}) | kWh/HDD | outdoor mean |".format(
            base=HDD_BASE
        )
    )
    print("|---|---|---|---|---|---|---|")
    print(
        "| {days} | {kwh:.0f} | {per_day:.1f} | {median:.1f} | {hdd:.0f} | {per_hdd} | {tout:.1f} |".format(
            days=summary["days"],
            kwh=summary["kwh_total"],
            per_day=summary["kwh_per_day"],
            median=summary["kwh_median"],
            hdd=summary["hdd_total"],
            per_hdd=_fmt(summary["kwh_per_hdd"]),
            tout=summary["outdoor_mean"],
        )
    )


# -------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-file")
    parser.add_argument("--start", help="ISO time (default: 30 days ago)")
    parser.add_argument("--end", help="ISO time (default: now)")
    parser.add_argument("--start-b", help="Optional second window start")
    parser.add_argument("--end-b", help="Optional second window end")
    parser.add_argument("--zones", help="Comma-separated zone ids (default: all found)")
    parser.add_argument(
        "--exclude", default="", help="Comma-separated zone ids to skip in window A"
    )
    parser.add_argument(
        "--exclude-b",
        help="Zone ids to skip in window B (default: same as --exclude)",
    )
    parser.add_argument("--comfort-min", type=float, default=21.0)
    parser.add_argument("--comfort-target", type=float, default=22.5)
    parser.add_argument("--comfort-max", type=float, default=24.0)
    parser.add_argument("--energy-entity", default=DEFAULT_ENERGY)
    parser.add_argument("--outdoor-entity", default=DEFAULT_OUTDOOR)
    parser.add_argument("--json-out", help="Write the raw metrics as JSON here")
    args = parser.parse_args(argv)

    token_file = (
        args.token_file
        or os.environ.get("HA_TOKEN_PATH")
        or os.path.join(os.environ.get("LOCALAPPDATA", ""), "ha_token.txt")
    )
    try:
        with open(token_file, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError as err:
        raise SystemExit(f"Cannot read the Home Assistant token: {err}") from err

    api = Api(args.base_url, token)
    now = datetime.now(timezone.utc)
    start = _parse_time(args.start) if args.start else now - timedelta(days=30)
    end = _parse_time(args.end) if args.end else now
    start_b = _parse_time(args.start_b) if args.start_b else None
    end_b = _parse_time(args.end_b) if args.end_b else now

    states = api.states()
    zones = discover_zones(states)
    if args.zones:
        wanted = {z.strip() for z in args.zones.split(",") if z.strip()}
        zones = {k: v for k, v in zones.items() if k in wanted}
    excluded_a = {z.strip() for z in args.exclude.split(",") if z.strip()}
    excluded_b = {
        z.strip()
        for z in (args.exclude_b if args.exclude_b is not None else args.exclude).split(",")
        if z.strip()
    }

    report: dict = {"windows": [], "generated_at": now.isoformat()}
    windows = [("Window A", start, end, excluded_a)]
    if start_b is not None:
        windows.append(("Window B", start_b, end_b, excluded_b))

    for title, win_start, win_end, excluded in windows:
        window_zones = {k: v for k, v in zones.items() if k not in excluded}
        rows = [
            zone_metrics(
                api,
                zone_id,
                zone,
                win_start,
                win_end,
                args.comfort_min,
                args.comfort_max,
                args.comfort_target,
            )
            for zone_id, zone in sorted(window_zones.items())
        ]
        energy = energy_summary(api, args.energy_entity, args.outdoor_entity, win_start, win_end)

        print(f"\n## {title}: {win_start.isoformat()} .. {win_end.isoformat()}")
        print_table(f"Comfort and control ({len(rows)} zones)", rows)
        print_energy("District heating vs outdoor temperature", energy)
        report["windows"].append(
            {
                "title": title,
                "start": win_start.isoformat(),
                "end": win_end.isoformat(),
                "zones": rows,
                "energy": energy,
            }
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote {args.json_out}")

    print(
        "\nNote: history is 10-minute sampled; kWh/HDD is indicative, not a "
        "billing figure. Windows are UTC."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
