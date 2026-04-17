"""
violation_logger.py  ·  AeroMRO Digital Twin  ·  v13 (Industry Grade)
══════════════════════════════════════════════════════════════════════
Outputs per-engine physics violation reports in MRO-grade format.
Each report contains:
  engine_id, cycle, top_violation, severity, ranked violations,
  recommended_action, predicted_rul_cycles, confidence_interval

Violation taxonomy (aligned to ATA Chapter 72 turbofan):
  COMPRESSOR — efficiency drop, HPC surge margin, fouling
  TURBINE    — TET exceedance, creep, TBC spallation
  STRUCTURAL — fatigue crack, disk burst risk, vibration
  THERMAL    — hot section temp exceedance, cooling degradation
  CORROSION  — environmental (coastal/desert/arctic routes)

This module is called after each few-shot eval loop.
Outputs: List[ViolationReport] + JSON log file.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch import Tensor


# ─── Violation taxonomy ───────────────────────────────────────────────────────

VIOLATION_DEFS = {
    # name → (feature_idx, threshold_normalised, ata_chapter, severity_base)
    "compressor_efficiency_drop": (17, 0.85, "ATA-72-30", "WARNING"),
    "turbine_efficiency_drop":    (18, 0.82, "ATA-72-50", "WARNING"),
    "fatigue_crack_growth":       (19, 0.60, "ATA-72-60", "CRITICAL"),
    "disk_burst_risk":            (20, 0.40, "ATA-72-60", "CRITICAL"),
    "fatigue_damage_high":        (14, 0.70, "ATA-72-60", "CAUTION"),
    "creep_damage_high":          (15, 0.65, "ATA-72-50", "CAUTION"),
    "corrosion_damage":           (16, 0.50, "ATA-72-30", "CAUTION"),
}

RECOMMENDED_ACTIONS = {
    "CRITICAL": "GROUND IMMEDIATELY — Borescope inspection required before next flight",
    "WARNING":  "Schedule maintenance within 5 flight cycles — reduced thrust operation",
    "CAUTION":  "Monitor closely — increased inspection interval recommended",
    "OK":       "Within normal parameters — continue standard maintenance schedule",
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ViolationItem:
    name:           str
    severity:       str       # CRITICAL / WARNING / CAUTION / OK
    ata_chapter:    str
    current_value:  float     # normalised [0,1]
    threshold:      float
    exceedance_pct: float     # how far above threshold (%)

@dataclass
class ViolationReport:
    engine_id:          int
    cycle:              int
    airline_id:         str
    route_type:         str
    predicted_rul:      float   # cycles
    rul_lower_95:       float   # 95% CI lower
    rul_upper_95:       float   # 95% CI upper
    health_index:       float
    top_violation:      str
    overall_severity:   str
    recommended_action: str
    violations:         List[ViolationItem] = field(default_factory=list)
    nasa_score:         float = 0.0
    rmse:               float = 0.0


# ─── Violation logger ─────────────────────────────────────────────────────────

class ViolationLogger:
    """
    Produces per-engine MRO violation reports from model outputs.

    Usage:
        logger = ViolationLogger(output_dir="logs/violations")
        for engine in eval_engines:
            report = logger.generate_report(engine_id, model_output, sensor_last)
            logger.log(report)
        logger.save_epoch_summary(epoch)
    """

    SEVERITY_RANK = {"CRITICAL": 4, "WARNING": 3, "CAUTION": 2, "OK": 1}

    def __init__(self, output_dir: str = "logs/violations"):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_reports: List[ViolationReport] = []

    def _check_violations(
        self, sensor_last: Tensor
    ) -> Tuple[List[ViolationItem], str, str]:
        """
        sensor_last: [26] — last timestep normalised features
        Returns: (violations_list, top_violation_name, overall_severity)
        """
        items: List[ViolationItem] = []
        s = sensor_last.float().cpu().numpy()

        for name, (feat_idx, threshold, ata, sev_base) in VIOLATION_DEFS.items():
            if feat_idx >= len(s):
                continue
            val = float(s[feat_idx])
            if val > threshold:
                exceed = (val - threshold) / (1.0 - threshold + 1e-8) * 100.0
                # Escalate severity if exceedance is severe (>50% above threshold)
                if exceed > 50.0 and sev_base == "CAUTION":
                    sev = "WARNING"
                elif exceed > 50.0 and sev_base == "WARNING":
                    sev = "CRITICAL"
                else:
                    sev = sev_base
                items.append(ViolationItem(
                    name=name, severity=sev, ata_chapter=ata,
                    current_value=round(val, 4),
                    threshold=threshold,
                    exceedance_pct=round(exceed, 2),
                ))

        if not items:
            return (
                [ViolationItem("all_parameters_nominal", "OK", "ATA-72",
                               0.0, 0.0, 0.0)],
                "all_parameters_nominal",
                "OK",
            )

        # Sort by severity rank, then exceedance
        items.sort(key=lambda x: (
            -self.SEVERITY_RANK[x.severity], -x.exceedance_pct
        ))
        top        = items[0].name
        overall    = items[0].severity
        return items, top, overall

    def generate_report(
        self,
        engine_id:     int,
        cycle:         int,
        airline_id:    str,
        route_type:    str,
        model_output:  Dict[str, Tensor],
        sensor_last:   Tensor,          # [26] last timestep
        true_rul:      Optional[float] = None,
    ) -> ViolationReport:
        """Generate a single-engine violation report from model output."""

        # RUL prediction
        rul_log    = float(model_output["rul_log"].squeeze().cpu().item())
        rul_cy     = float(math.expm1(max(0.0, rul_log)))

        # Confidence interval from uncertainty
        log_var    = float(model_output["rul_log_var"].squeeze().cpu().item())
        log_var    = max(-4.0, min(6.0, log_var))
        std_log    = math.exp(0.5 * log_var)
        # 95% CI in log-space, back-transformed
        lower_log  = max(0.0, rul_log - 1.96 * std_log)
        upper_log  = rul_log + 1.96 * std_log
        rul_lower  = float(math.expm1(lower_log))
        rul_upper  = float(math.expm1(upper_log))

        # Health
        health = float(model_output["health"].squeeze().cpu().item())

        # Physics violations
        violations, top_viol, overall_sev = self._check_violations(sensor_last)

        # NASA score if true RUL available
        nasa = 0.0
        rmse = 0.0
        if true_rul is not None:
            err  = rul_cy - true_rul
            d    = max(-50.0, min(50.0, err))
            nasa = math.exp(d / 10.0) - 1.0 if d >= 0 else math.exp(-d / 13.0) - 1.0
            rmse = abs(err)

        return ViolationReport(
            engine_id          = engine_id,
            cycle              = cycle,
            airline_id         = airline_id,
            route_type         = route_type,
            predicted_rul      = round(rul_cy, 1),
            rul_lower_95       = round(rul_lower, 1),
            rul_upper_95       = round(rul_upper, 1),
            health_index       = round(health, 4),
            top_violation      = top_viol,
            overall_severity   = overall_sev,
            recommended_action = RECOMMENDED_ACTIONS[overall_sev],
            violations         = violations,
            nasa_score         = round(nasa, 2),
            rmse               = round(rmse, 2),
        )

    def log(self, report: ViolationReport) -> None:
        self._epoch_reports.append(report)

    def save_epoch_summary(self, epoch: int) -> str:
        """
        Write JSON summary for this epoch.
        Returns path to the written file.
        """
        path = self.output_dir / f"violations_epoch_{epoch:04d}.json"

        critical = [r for r in self._epoch_reports
                    if r.overall_severity == "CRITICAL"]
        warning  = [r for r in self._epoch_reports
                    if r.overall_severity == "WARNING"]

        summary = {
            "epoch":            epoch,
            "total_engines":    len(self._epoch_reports),
            "critical_count":   len(critical),
            "warning_count":    len(warning),
            "mean_rul":         float(np.mean([r.predicted_rul for r in self._epoch_reports]))
                                if self._epoch_reports else 0.0,
            "mean_nasa":        float(np.mean([r.nasa_score for r in self._epoch_reports]))
                                if self._epoch_reports else 0.0,
            "mean_rmse":        float(np.mean([r.rmse for r in self._epoch_reports]))
                                if self._epoch_reports else 0.0,
            "top_violations":   self._top_violations_summary(),
            "reports":          [asdict(r) for r in self._epoch_reports],
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        self._epoch_reports = []  # reset for next epoch
        return str(path)

    def print_critical_alerts(self) -> None:
        """Print CRITICAL violations to console — for real-time monitoring."""
        for r in self._epoch_reports:
            if r.overall_severity == "CRITICAL":
                print(
                    f"  ⚠ CRITICAL | Engine {r.engine_id:>5} | "
                    f"Cycle {r.cycle:>4} | RUL={r.predicted_rul:.0f}cy | "
                    f"{r.top_violation}"
                )

    def _top_violations_summary(self) -> Dict[str, int]:
        """Frequency count of top violations across all engines this epoch."""
        counts: Dict[str, int] = {}
        for r in self._epoch_reports:
            counts[r.top_violation] = counts.get(r.top_violation, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:10])