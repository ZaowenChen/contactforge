"""Explicit clearance transfer; geometry-only revision-4 outcome reporting."""
from collections import Counter
import math
import numpy as np
from .config import BlindConfig, ASSET_CONTRACT_KEYS


def clearance_transfer_contract(training_clearance, test_clearance, enabled):
    changed = not math.isclose(training_clearance,test_clearance,rel_tol=0.,abs_tol=1e-12)
    if enabled:
        if not (math.isclose(training_clearance,.0005,rel_tol=0.,abs_tol=1e-12)
                and math.isclose(test_clearance,.00005,rel_tol=0.,abs_tol=1e-12)):
            raise ValueError("clearance transfer requires 0.5 mm -> 0.05 mm")
    elif changed:
        raise ValueError("evaluation clearance differs; explicit clearance transfer required")
    return dict(evaluation_domain="clearance_transfer" if enabled else "training_clearance",
                training_clearance_per_side_m=training_clearance,test_clearance_per_side_m=test_clearance,
                zero_shot=bool(enabled),allowed_contract_changes=["clearance_per_side_m"] if enabled else [])


def check_asset_contract(source, cfg):
    defaults=BlindConfig().to_dict();current=cfg.to_dict()
    changed=[key for key in ASSET_CONTRACT_KEYS if source.get(key,defaults[key]) != current[key]]
    if changed:
        raise ValueError(f"checkpoint asset contract differs: {changed}")


def check_initialization_contract(source, cfg):
    """Only the declared A-to-B curriculum may change the task on initialization."""
    check_asset_contract(source,cfg)
    defaults=BlindConfig().to_dict()
    allowed={"seed","learning_rate","clearance_per_side_m"}
    changed=[key for key,value in cfg.to_dict().items()
             if source.get(key,defaults.get(key)) != value and key not in allowed]
    if changed:
        raise ValueError(f"initialization contract differs: {changed}")
    if source["clearance_per_side_m"] != cfg.clearance_per_side_m:
        clearance_transfer_contract(source["clearance_per_side_m"],cfg.clearance_per_side_m,True)


def quantiles(values):
    return dict(zip(("min","p05","p50","p95","p99","max"),
                    np.quantile(values,(0,.05,.5,.95,.99,1)).tolist()))


def summarize_cases(rows,cfg):
    reasons=Counter()
    tolerance=min(cfg.geometry_tolerance_m,cfg.clearance_per_side_m/10.)
    strict=0
    for row in rows:
        success=row["outcome"]=="success"
        row["strict_geometry_success"]=bool(success and row["shaft_error_m"]<=tolerance)
        strict += row["strict_geometry_success"]
        reason=(None if success else "force_and_moment_limit" if row["fault_code"] & 3 == 3
                else "force_limit" if row["fault_code"] & 1 else "moment_limit" if row["fault_code"] & 2
                else "far_from_fixture" if row["fault_code"] & 4
                else "timeout_after_capture" if "capture_time_s" in row else "timeout_without_capture")
        row["failure_reason"]=reason
        if reason: reasons[reason]+=1
    result = dict(depth_at_end_m_quantiles=quantiles([r["depth_m"] for r in rows]),
                episode_max_depth_m_quantiles=quantiles([r["max_depth_m"] for r in rows]),
                episode_peak_force_n_quantiles=quantiles([r["force_peak_n"] for r in rows]),
                failure_reasons=dict(reasons),strict_geometry_tolerance_m=tolerance,
                strict_geometry_successes=strict,strict_geometry_success_rate=strict/len(rows),
                strict_geometry_scope=("supplementary terminal shaft tolerance; chamfer-consistent seat" if cfg.seating_protocol=="chamfer_v1"
                                       else "supplementary terminal geometry; unchanged v4 seating protocol"),
                force_role=("1 kHz unfiltered force/moment peaks gate success and latch load faults" if cfg.study_revision==6
                            else "diagnostic only, sampled at policy rate; no force-qualified success criterion"))

    if cfg.study_revision == 6:
        result.update(geometry_successes=sum(r["geometry_success"] for r in rows),
                      force_qualified_successes=sum(r["force_qualified_success"] for r in rows),
                      episode_peak_moment_nm_quantiles=quantiles([r["moment_peak_nm"] for r in rows]),
                      force_exceed_s_total=sum(r["force_exceed_s"] for r in rows),
                      moment_exceed_s_total=sum(r["moment_exceed_s"] for r in rows))
    return result
