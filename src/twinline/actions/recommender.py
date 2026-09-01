"""build_recommendation(): turns a rootcause driver + containment list into
an advisory action — shadow mode only, never a PLC write. expected_impact is
always computed from the actual units-at-risk count, a caller-supplied
probability estimate, and the configured rework cost; the lever text is
templated with the real station/shift/count, never a hardcoded string.
"""

from twinline.schemas import ActionsConfig, Recommendation, UnitsAtRisk

_OWNER_BY_DRIVER = {
    "tool_wear": "maintenance_lead",
    "supplier_batch": "quality",
    "operator_variation": "shift_supervisor",
    "ambient": "facilities",
    "bottleneck": "plant_manager",
}

_LEVER_BY_DRIVER = {
    "tool_wear": "schedule_tool_change",
    "supplier_batch": "quarantine_batch",
    "operator_variation": "pair_check_and_refresh",
    "ambient": "raise_dehumidification_setpoint",
    "bottleneck": "add_buffer_or_rebalance",
}


def build_recommendation(
    driver: str,
    station_id: str,
    units_at_risk: UnitsAtRisk,
    probability: float,
    confidence: float,
    rework_cost: float,
    cfg: ActionsConfig,
    shift_id: str | None = None,
) -> Recommendation:
    if driver not in _OWNER_BY_DRIVER:
        raise ValueError(f"no lever defined for driver {driver!r}")

    n_at_risk = len(units_at_risk.unit_ids)
    expected_impact = probability * n_at_risk * rework_cost
    action, monitoring_plan, requires_window = _action_text(driver, station_id, n_at_risk, shift_id, cfg)

    return Recommendation(
        driver=driver, controllable_lever=_LEVER_BY_DRIVER[driver], action=action,
        expected_impact=expected_impact, impact_units="currency avoided",
        owner_role=_OWNER_BY_DRIVER[driver], confidence=confidence, monitoring_plan=monitoring_plan,
        requires_maintenance_window=requires_window,
    )


def _action_text(
    driver: str, station_id: str, n_at_risk: int, shift_id: str | None, cfg: ActionsConfig
) -> tuple[str, str, bool]:
    window = cfg.monitoring_units_window

    if driver == "tool_wear":
        return (
            f"Schedule a tool change at {station_id} within {cfg.tool_wear_change_window_hours:g}h.",
            f"After the change, confirm {station_id}'s process sensors stay within control limits for the next {window} units.",
            True,
        )
    if driver == "supplier_batch":
        return (
            f"Quarantine the current supplier batch feeding {station_id} and inspect the {n_at_risk} at-risk units.",
            f"Hold incoming material at {station_id} until the quarantined batch clears inspection.",
            False,
        )
    if driver == "operator_variation":
        shift_clause = f" on {shift_id}" if shift_id else ""
        return (
            f"Run a pair-check and refresher on the {station_id} step with the operator{shift_clause}.",
            f"Track {station_id}'s manual check pass rate for the next {window} units after the refresher.",
            False,
        )
    if driver == "ambient":
        return (
            f"Raise the paint booth dehumidification setpoint near {station_id}.",
            f"Watch {station_id}'s finish-quality signal for the next {window} units after adjusting.",
            False,
        )
    if driver == "bottleneck":
        return (
            f"Add buffer slots or rebalance work content around {station_id}.",
            f"Monitor {station_id}'s blocked/starved ratio over the next {window} units for improvement.",
            True,
        )
    raise ValueError(f"no lever defined for driver {driver!r}")
