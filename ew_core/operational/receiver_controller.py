"""
End-to-End Operational Receiver Controller for Cognitive EW Scanning.

Architectural Contract:
    MissionClock -> OperationalStateBuilder (360-D) -> SmartScanMoE (Frozen v2) ->
    ReceiverAdapter (Retune + Dwell) -> Physical RF Extraction ->
    EmitterTracker (Unsupervised Association) -> Temporal & Spatial Feedback -> Telemetry

CRITICAL GOVERNANCE RULES:
    1. The controller MUST NOT contain an independent scheduling policy.
       Scheduling authority is strictly:
           TemporalPredictor + SpatialTracker + v2 DRQN Scheduler
    2. ZERO dependence on ground-truth emitter IDs during live operation.
       Interception, association, and tracking operate purely on measured physical attributes:
           (frequency_mhz, time_us, pulse_width_us, amplitude_db, aoa_deg)
       Ground-truth emitter labels are restricted to offline simulation verification.
    3. Single authoritative MissionClock across all receiver, scheduling, and tracking layers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    DWELL_MODE_SEMANTICS,
    band_of_action,
    mode_of_action,
    validate_action,
)
from ew_core.cognitive.spatial_tracker import SpatialTracker
from ew_core.cognitive.temporal_predictor import TemporalPredictor
from ew_core.models.smartscan_moe import SmartScanMoE
from ew_core.perception.emitter_tracker import EmitterTracker
from ew_core.receiver.mission_clock import MissionClock
from ew_core.receiver.models import DetectionObservation, ReceiverObservation
from ew_core.receiver.sieve_receiver import SieveReceiver
from ew_core.operational.receiver_adapter import ReceiverAdapter, ReceiverHardwareError
from ew_core.operational.state_builder import OperationalStateBuilder

logger = logging.getLogger(__name__)


@dataclass
class ReceiverTelemetryFrame:
    """Standardized operational telemetry frame for logging, APIs, and dashboard."""

    step: int
    timestamp_us: float
    dwell_start_us: float
    dwell_end_us: float
    selected_band: int
    selected_mode: int
    mode_name: str
    center_frequency_mhz: float
    bandwidth_mhz: float
    dwell_duration_us: float
    retune_latency_us: float

    # Outcome
    hit: bool
    num_detections: int
    intercept_time_us: Optional[float]
    detections: List[Dict[str, Any]] = field(default_factory=list)
    band_priorities: List[float] = field(default_factory=list)
    observation: List[float] = field(default_factory=list)

    # Cognitive Explanation ("WHY THIS BAND?")
    decision_reason: str = "unknown"
    predicted_track_id: str = "None"
    predicted_band: int = -1
    p_next_band: float = 0.0
    predicted_eta_us: float = -1.0
    spatial_confidence: float = 0.0
    aoa_deg: float = -1.0
    agility_score: float = 0.0
    prediction_confidence: float = 0.0

    # Arbitration component scores
    drqn_score: float = 0.0
    predictive_score: float = 0.0
    spatial_score: float = 0.0
    exploration_pressure: float = 0.0
    q_margin: float = 0.0

    # Policy, arbitration, and override diagnostics
    policy_mode: str = "operational"
    exploration_enabled: bool = False
    action_was_overridden: bool = False
    fallback_triggered: float = 0.0
    fallback_reason: str = "none"
    action_rejection_reason: str = "none"
    drqn_candidate_active: float = 1.0
    confidence_threshold: float = 0.020
    drqn_candidate_score: float = 0.0
    arbitration_score: float = 0.0
    operational_checkpoint: str = "Gate-25k-R4.2-alpha020"

    # System rolling figures of merit
    rolling_pd: float = 0.0
    rolling_median_latency_us: float = 0.0
    consecutive_empty_band: int = 0
    consecutive_empty_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_us": self.timestamp_us,
            "dwell_start_us": self.dwell_start_us,
            "dwell_end_us": self.dwell_end_us,
            "selected_band": self.selected_band,
            "selected_mode": self.selected_mode,
            "mode_name": self.mode_name,
            "center_frequency_mhz": self.center_frequency_mhz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "dwell_duration_us": self.dwell_duration_us,
            "retune_latency_us": self.retune_latency_us,
            "hit": self.hit,
            "num_detections": self.num_detections,
            "intercept_time_us": self.intercept_time_us,
            "detections": self.detections,
            "band_priorities": self.band_priorities,
            "observation": self.observation,
            "cognitive_explanation": {
                "decision_reason": self.decision_reason,
                "predicted_track_id": self.predicted_track_id,
                "predicted_band": self.predicted_band,
                "p_next_band": self.p_next_band,
                "predicted_eta_us": self.predicted_eta_us,
                "spatial_confidence": self.spatial_confidence,
                "aoa_deg": self.aoa_deg,
                "agility_score": self.agility_score,
                "prediction_confidence": self.prediction_confidence,
                "drqn_score": self.drqn_score,
                "predictive_score": self.predictive_score,
                "spatial_score": self.spatial_score,
                "exploration_pressure": self.exploration_pressure,
                "q_margin": self.q_margin,
                "policy_mode": self.policy_mode,
                "exploration_enabled": self.exploration_enabled,
                "action_was_overridden": self.action_was_overridden,
                "fallback_triggered": self.fallback_triggered,
                "fallback_reason": self.fallback_reason,
                "action_rejection_reason": self.action_rejection_reason,
                "drqn_candidate_active": self.drqn_candidate_active,
                "confidence_threshold": self.confidence_threshold,
                "drqn_candidate_score": self.drqn_candidate_score,
                "arbitration_score": self.arbitration_score,
                "operational_checkpoint": self.operational_checkpoint,
            },
            "system_metrics": {
                "rolling_pd": self.rolling_pd,
                "rolling_median_latency_us": self.rolling_median_latency_us,
                "consecutive_empty_band": self.consecutive_empty_band,
                "consecutive_empty_total": self.consecutive_empty_total,
            },
        }


class OperationalReceiverController:
    """Closed-Loop Operational Receiver Controller & Mission Orchestrator.

    Owns the full operational loop without simulation shortcuts or oracle truth.
    """

    ROLLING_WINDOW_SIZE: int = 100

    def __init__(
        self,
        scheduler: Any = None,
        moe_scheduler: Optional[Any] = None,
        receiver_adapter: Optional[ReceiverAdapter] = None,
        clock: Optional[MissionClock] = None,
        state_builder: Optional[OperationalStateBuilder] = None,
        emitter_tracker: Optional[EmitterTracker] = None,
        retune_latency_us: float = 15.0,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
    ) -> None:
        self.scheduler = scheduler if scheduler is not None else moe_scheduler
        self.moe_scheduler = self.scheduler  # Backward compatibility alias
        self.n_bands = int(n_bands)
        self.n_modes = int(n_modes)
        self.retune_latency_us = float(retune_latency_us)

        # Core runtime components
        self.clock = clock or MissionClock()
        self.receiver_adapter = receiver_adapter or ReceiverAdapter()
        self.state_builder = state_builder or OperationalStateBuilder(n_bands=self.n_bands)
        self.emitter_tracker = emitter_tracker or EmitterTracker(n_bands=self.n_bands)

        # Operational state
        self.current_step: int = 0
        self.is_mission_active: bool = False
        self.pdw_buffer: List[Dict[str, Any]] = []
        self.telemetry_history: List[ReceiverTelemetryFrame] = []

        # Rolling statistics
        self.total_dwells: int = 0
        self.total_hits: int = 0
        self.latencies: List[float] = []

    @property
    def clock_us(self) -> float:
        """Compatibility property for mission clock."""
        return self.clock.current_time_us

    @clock_us.setter
    def clock_us(self, val: float) -> None:
        self.clock.reset(float(val))

    def reset(self, initial_time_us: float = 0.0) -> None:
        """Reset internal receiver, mission clock, and tracking states."""
        self.current_step = 0
        self.is_mission_active = False
        self.clock.reset(initial_time_us)
        self.pdw_buffer.clear()
        self.telemetry_history.clear()
        self.total_dwells = 0
        self.total_hits = 0
        self.latencies.clear()

        self.receiver_adapter.reset()
        self.state_builder.reset()
        self.emitter_tracker.reset()
        if hasattr(self.scheduler, "reset"):
            self.scheduler.reset()

    def start_mission(self, initial_time_us: float = 0.0) -> None:
        """Commence operational closed-loop mission."""
        self.reset(initial_time_us)
        self.is_mission_active = True
        logger.info("[OPERATIONAL CONTROLLER] Mission started at t = %.1f µs", initial_time_us)

    def stop_mission(self) -> None:
        """Terminate operational mission."""
        self.is_mission_active = False
        logger.info("[OPERATIONAL CONTROLLER] Mission stopped after %d dwells (%d hits)", self.total_dwells, self.total_hits)

    def band_to_center_freq(self, band: int) -> float:
        """Map band index [0, 35] to receiver center frequency in MHz."""
        return float(band * 500.0 + 250.0)

    def mode_to_duration(self, mode: int) -> float:
        """Map dwell mode [0, 4] to physical dwell duration in µs."""
        mult = DEFAULT_DWELL_MULTIPLIERS[mode] if mode < len(DEFAULT_DWELL_MULTIPLIERS) else 1.0
        return float(500.0 * mult)

    def ingest_pdw(
        self,
        time_us: float,
        frequency_mhz: float,
        pulse_width_us: float = 1.0,
        amplitude_db: float = -50.0,
        aoa_deg: float = 0.0,
        source_pulse_id: Optional[Any] = None,
        ground_truth_emitter_id: Optional[int] = None,
    ) -> int:
        """Ingest external PDW into perception pipeline WITHOUT privileged truth.
        
        Associates pulse via EmitterTracker to determine internal associated_track_id.
        
        Returns:
            The associated internal track ID.
        """
        # Validate input parameters against corruption / NaNs
        if not math.isfinite(float(time_us)) or float(time_us) < 0.0:
            raise ValueError(f"Invalid pulse time_us: {time_us!r}")
        if not math.isfinite(float(frequency_mhz)) or float(frequency_mhz) < 0.0 or float(frequency_mhz) > 18000.0:
            raise ValueError(f"Invalid pulse frequency_mhz: {frequency_mhz!r}")
        if not math.isfinite(float(pulse_width_us)) or float(pulse_width_us) <= 0.0:
            raise ValueError(f"Invalid pulse_width_us: {pulse_width_us!r}")
        if not math.isfinite(float(amplitude_db)):
            raise ValueError(f"Invalid amplitude_db: {amplitude_db!r}")

        band = int(min(self.n_bands - 1, max(0, int(frequency_mhz // 500.0))))
        clean_aoa = float(aoa_deg) if math.isfinite(float(aoa_deg)) else 0.0

        pdw = {
            "time_us": float(time_us),
            "frequency_mhz": float(frequency_mhz),
            "pulse_width_us": float(pulse_width_us),
            "amplitude_db": float(amplitude_db),
            "aoa_deg": clean_aoa,
            "pulse_id": source_pulse_id,
            "band": band,
        }
        self.pdw_buffer.append(pdw)

        # Associate pulse into internal tracks
        track_id = self._associate_pulse_to_track(pdw)

        # Update causal tracking layers using internal track_id
        self.moe_scheduler.temporal_predictor.update_from_pulse(
            track_id=track_id,
            toa_us=pdw["time_us"],
            freq_mhz=pdw["frequency_mhz"],
            band=pdw["band"],
        )
        if math.isfinite(clean_aoa):
            self.moe_scheduler.spatial_tracker.update_from_track(
                track_id=track_id,
                new_aoa_deg=clean_aoa,
                current_time_us=pdw["time_us"],
            )

        return track_id

    def _associate_pulse_to_track(self, pdw: Dict[str, Any]) -> int:
        """Associate a single physical PDW to an existing or new emitter track."""
        f_pulse = float(pdw["frequency_mhz"])
        aoa_pulse = float(pdw.get("aoa_deg", 0.0))
        t_pulse = float(pdw["time_us"])
        band = int(pdw.get("band", min(self.n_bands - 1, max(0, int(f_pulse // 500.0)))))

        labels = np.array([0], dtype=np.int64)
        toa_us = np.array([t_pulse], dtype=np.float64)
        freq_mhz = np.array([f_pulse], dtype=np.float64)
        aoa_deg = np.array([aoa_pulse], dtype=np.float64)
        pw_us = np.array([pdw["pulse_width_us"]], dtype=np.float64)
        amp_db = np.array([pdw["amplitude_db"]], dtype=np.float64)

        updated_tracks = self.emitter_tracker.update_from_deinterleaver(
            labels=labels,
            toa_us=toa_us,
            freq_mhz=freq_mhz,
            aoa_deg=aoa_deg,
            pw_us=pw_us,
            amp_db=amp_db,
            current_time=t_pulse,
            band=band,
            min_cluster_size=1,
        )

        assigned_tids = self.emitter_tracker.get_pulse_track_assignment(labels)
        if len(assigned_tids) > 0 and assigned_tids[0] >= 0:
            return int(assigned_tids[0])
        elif updated_tracks:
            return int(next(iter(updated_tracks.keys())))
        return 0

    def execute_operational_step(
        self,
        obs: Optional[np.ndarray] = None,
        scenario_pulses: Optional[Sequence[Any]] = None,
        external_rf_stream: Optional[Sequence[Any]] = None,
    ) -> ReceiverTelemetryFrame:
        """Execute one complete, closed-loop operational scan cycle.
        
        Flow:
          1. Synchronize Mission Clock
          2. Causally buffer incident RF (if pulses provided)
          3. Build canonical 360-D observation state (or use supplied obs)
          4. SmartScanMoE cognitive arbitration (action = b * 5 + m)
          5. Advance clock by retune latency and tune receiver
          6. Advance clock by dwell duration and execute physical aperture
          7. Unsupervised pulse association (no ground-truth emitter_id)
          8. Feedback to trackers and state builder
          9. Publish standardized telemetry frame
        """
        t_now = self.clock.current_time_us

        # 1. Ingest incident RF causally into receiver front-end
        stream = external_rf_stream if external_rf_stream is not None else scenario_pulses
        if stream is not None:
            # Allow feeding up to generous dwell lookahead (e.g. 15 us retune + 1250 us max dwell)
            self.receiver_adapter.feed_incident_rf(
                stream,
                max_time_us=t_now + self.retune_latency_us + 1500.0,
            )

        # 2. Build canonical 360-D observation state
        if obs is not None:
            effective_obs = obs
        else:
            effective_obs = self.state_builder.build_state(
                current_time_us=t_now,
                active_tracks=self.emitter_tracker.tracks,
                spatial_tracker=getattr(self.scheduler, "spatial_tracker", None),
                temporal_predictor=getattr(self.scheduler, "temporal_predictor", None),
            )

        # 3. Schedule action via scheduler (DRQN or MoE)
        if hasattr(self.scheduler, "_simulated_clock_us"):
            self.scheduler._simulated_clock_us = t_now
        if hasattr(self.scheduler, "select_action"):
            action, _, attr = self.scheduler.select_action(effective_obs)
        elif hasattr(self.scheduler, "act"):
            action, attr = self.scheduler.act(effective_obs)
        else:
            action = self.scheduler.step(effective_obs)
            attr = {}
        attr = attr or {}
        action = validate_action(action, n_actions=self.n_bands * self.n_modes)
        selected_band = band_of_action(action, self.n_modes)
        selected_mode = mode_of_action(action, self.n_modes)
        dwell_duration = self.mode_to_duration(selected_mode)

        # 4. Receiver Actuation: Retune
        target_center_mhz = self.band_to_center_freq(selected_band)
        self.receiver_adapter.tune(target_center_mhz)
        self.receiver_adapter.set_dwell_time(dwell_duration)
        retune_start, retune_end = self.clock.advance_retune(self.retune_latency_us)

        # 5. Dwell Execution: Physical Aperture Window
        dwell_start, dwell_end = self.clock.advance_dwell(dwell_duration)
        detected_pdws = self.receiver_adapter.execute_dwell(dwell_start, dwell_end)

        hit = len(detected_pdws) > 0
        intercept_time_us: Optional[float] = None
        if hit:
            t_first = min(p["time_us"] for p in detected_pdws)
            intercept_time_us = float(max(0.0, t_first - dwell_start))
            self.latencies.append(intercept_time_us)
            self.total_hits += 1

        self.total_dwells += 1

        # 6. Unsupervised Track Association & Perception Feedback
        if hit:
            # Associate detected pulses without ground-truth labels
            for p in detected_pdws:
                t_arr = p["time_us"]
                f_pulse = p["frequency_mhz"]
                aoa = p["aoa_deg"]
                tid = self._associate_pulse_to_track(p)

                # Feed into causal predictors if present
                if hasattr(self.scheduler, "temporal_predictor"):
                    self.scheduler.temporal_predictor.update_from_pulse(
                        track_id=tid,
                        toa_us=t_arr,
                        freq_mhz=f_pulse,
                        band=selected_band,
                    )
                if math.isfinite(aoa) and hasattr(self.scheduler, "spatial_tracker"):
                    self.scheduler.spatial_tracker.update_from_track(
                        track_id=tid,
                        new_aoa_deg=aoa,
                        current_time_us=t_arr,
                    )

        # Feedback dwell outcome to state builder and scheduler
        self.state_builder.record_dwell_outcome(
            band=selected_band,
            hit=hit,
            current_time_us=dwell_end,
            num_pulses=len(detected_pdws),
        )
        if hasattr(self.scheduler, "update_result"):
            try:
                self.scheduler.update_result(
                    hit=hit,
                    band=selected_band,
                    detections=detected_pdws,
                    current_time=dwell_end,
                )
            except TypeError:
                self.scheduler.update_result(hit=hit, band=selected_band)

        if hasattr(self.scheduler, "update"):
            self.scheduler.update(action)

        # 7. Construct Standardized Telemetry Frame
        rolling_pd = float(self.total_hits / self.total_dwells) if self.total_dwells > 0 else 0.0
        rolling_med_lat = float(np.median(self.latencies[-self.ROLLING_WINDOW_SIZE:])) if self.latencies else 0.0

        eff_flat = np.asarray(effective_obs, dtype=np.float32).flatten() if effective_obs is not None else np.zeros(0, dtype=np.float32)
        if len(eff_flat) >= self.n_bands * 10:
            band_priorities = [float(eff_flat[b * 10]) for b in range(self.n_bands)]
        else:
            band_priorities = [0.0] * self.n_bands
        obs_vec = eff_flat.tolist() if len(eff_flat) == self.n_bands * 10 else []

        telemetry_frame = ReceiverTelemetryFrame(
            step=self.current_step,
            timestamp_us=dwell_end,
            dwell_start_us=dwell_start,
            dwell_end_us=dwell_end,
            selected_band=selected_band,
            selected_mode=selected_mode,
            mode_name=DWELL_MODES[selected_mode],
            center_frequency_mhz=target_center_mhz,
            bandwidth_mhz=500.0,
            dwell_duration_us=dwell_duration,
            retune_latency_us=self.retune_latency_us,
            hit=hit,
            num_detections=len(detected_pdws),
            intercept_time_us=intercept_time_us,
            detections=detected_pdws,
            band_priorities=band_priorities,
            observation=obs_vec,
            decision_reason=str(attr.get("reason", attr.get("decision_reason", "DRQN_active"))),
            predicted_track_id=str(attr.get("predicted_track_id", "None")),
            predicted_band=int(attr.get("predicted_band", -1)),
            p_next_band=float(attr.get("p_next_band", 0.0)),
            predicted_eta_us=float(max(0.0, float(attr.get("predicted_eta_us", attr.get("eta_us", 0.0))))),
            spatial_confidence=float(max(0.0, float(attr.get("spatial_confidence", 0.0)))),
            aoa_deg=float(attr.get("aoa_deg", -1.0)),
            agility_score=float(max(0.0, float(attr.get("agility_score", 0.0)))),
            prediction_confidence=float(max(0.0, float(attr.get("prediction_confidence", 0.0)))),
            drqn_score=float(attr.get("drqn_score", attr.get("action_score", 0.0))),
            predictive_score=float(attr.get("predictive_score", 0.0)),
            spatial_score=float(attr.get("spatial_score", 0.0)),
            exploration_pressure=float(max(0.0, float(attr.get("exploration_pressure", attr.get("revisit_pct", 0.0))))),
            q_margin=float(attr.get("q_margin", 0.0)),
            rolling_pd=rolling_pd,
            rolling_median_latency_us=rolling_med_lat,
            consecutive_empty_band=int(getattr(self.moe_scheduler, "_consecutive_empty_band", 0)),
            consecutive_empty_total=int(getattr(self.moe_scheduler, "_consecutive_empty_total", 0)),
            policy_mode=str(attr.get("policy_mode", "operational")),
            exploration_enabled=bool(attr.get("exploration_enabled", False)),
            action_was_overridden=bool(attr.get("action_was_overridden", False)),
            fallback_triggered=float(attr.get("fallback_triggered", 0.0)),
            fallback_reason=str(attr.get("fallback_reason", "none")),
            action_rejection_reason=str(attr.get("action_rejection_reason", "none")),
            drqn_candidate_active=float(attr.get("drqn_candidate_active", 1.0)),
            confidence_threshold=float(attr.get("confidence_threshold", 0.020)),
            drqn_candidate_score=float(attr.get("drqn_candidate_score", 0.0)),
            arbitration_score=float(attr.get("arbitration_score", 0.0)),
            operational_checkpoint=str(attr.get("operational_checkpoint", "Gate-25k-R4.2-alpha020")),
        )

        self.telemetry_history.append(telemetry_frame)
        self.current_step += 1
        return telemetry_frame

    def step(
        self,
        obs: Optional[np.ndarray] = None,
        scenario_pulses: Optional[Sequence[Any]] = None,
        external_rf_stream: Optional[Sequence[Any]] = None,
    ) -> ReceiverTelemetryFrame:
        """Execute one complete, closed-loop operational scan cycle.

        Alias for execute_operational_step for controller interface consistency.
        """
        return self.execute_operational_step(
            obs=obs,
            scenario_pulses=scenario_pulses,
            external_rf_stream=external_rf_stream,
        )
