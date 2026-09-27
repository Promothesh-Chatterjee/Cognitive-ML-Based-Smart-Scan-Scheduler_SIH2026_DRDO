"""Phase G8.1 Fail-Closed Stratified Replay Buffer Sampler.

Implements authoritative, mutually exclusive 4-stratum sequence replay:
Precedence Order: agile > sparse > dense > mixed

Strata Mapping:
  - Agile: config_29, config_119, config_241 (agile hoppers; agile takes precedence over sparse for config_119)
  - Sparse: config_143 (and pure sparse non-hopping emitters)
  - Dense: config_194, config_195 (continuous / high emitter density stare)
  - Mixed: config_117, config_64, config_96, config_42 (periodic, cyclic, fixed, etc.)

Sampling Proportions:
  - Agile: 30%
  - Sparse: 25%
  - Dense: 25%
  - Mixed: 20%

G8.1 INVARIANT: FAIL-CLOSED.
If any stratum has fewer than min_episodes_per_stratum (default 5) or insufficient transitions,
this sampler raises RuntimeError. It NEVER silently falls back to uniform replay.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.scenario_classifier import classify_scenario

logger = logging.getLogger(__name__)

# Canonical scenario-to-stratum mapping with authoritative precedence: agile > sparse > dense > mixed
CANONICAL_SCENARIO_STRATA: dict[str, str] = {
    # Agile (Precedence 1 - agile hoppers)
    "config_29": "agile",
    "config_119": "agile",  # Authoritative: agile takes precedence over sparse
    "config_241": "agile",
    # Sparse (Precedence 2 - pure sparse non-hopping)
    "config_143": "sparse",
    "config_1000": "sparse",
    "config_1001": "sparse",
    "config_1004": "sparse",
    # Dense (Precedence 3 - high emitter density / continuous stare)
    "config_194": "dense",
    "config_195": "dense",
    # Mixed (Precedence 4 - periodic, cyclic, fixed, etc.)
    "config_117": "mixed",
    "config_64": "mixed",
    "config_96": "mixed",
    "config_42": "mixed",
}

VALID_STRATA = ("agile", "sparse", "dense", "mixed")


def map_scenario_to_primary_stratum(scenario_id: str, scenario_class: Optional[str] = None) -> str:
    """Map scenario ID and class to exactly one mutually exclusive primary stratum.

    Precedence order: agile > sparse > dense > mixed.
    """
    scen_id_clean = str(scenario_id).strip().lower().split(":")[-1]
    # Remove file extension if present
    if scen_id_clean.endswith(".h5"):
        scen_id_clean = scen_id_clean[:-3]

    if scen_id_clean in CANONICAL_SCENARIO_STRATA:
        return CANONICAL_SCENARIO_STRATA[scen_id_clean]

    # Precedence fallback by scenario class
    cls_str = str(scenario_class or classify_scenario(scen_id_clean)).strip().lower()
    if "agile" in cls_str:
        return "agile"
    if "sparse" in cls_str:
        return "sparse"
    if "dense" in cls_str:
        return "dense"
    return "mixed"


class StratifiedReplaySampler:
    """4-Stratum balanced sequence sampler wrapping SequenceReplayBuffer with fail-closed guarantee."""

    def __init__(
        self,
        buffer: SequenceReplayBuffer,
        strata_weights: Optional[Dict[str, float]] = None,
        min_episodes_per_stratum: int = 5,
        targeted_spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.buffer = buffer
        if strata_weights is None:
            strata_weights = {
                "agile": 0.30,
                "sparse": 0.25,
                "dense": 0.25,
                "mixed": 0.20,
            }
        self.strata_weights = strata_weights
        self.min_episodes_per_stratum = int(min_episodes_per_stratum)
        self.targeted_spec = targeted_spec

    def _is_targeted_episode(self, ep: dict) -> bool:
        """Check if an episode belongs to the targeted scenario-mode pool."""
        if not self.targeted_spec:
            return False
        target_scen = self.targeted_spec.get("scenario_id")
        target_mode = self.targeted_spec.get("mode")

        scen_id = str(ep.get("scenario_id", "")).strip().lower().split(":")[-1]
        if scen_id.endswith(".h5"):
            scen_id = scen_id[:-3]

        if target_scen is not None and scen_id != str(target_scen).strip().lower():
            return False

        if target_mode is not None:
            if ep.get("target_mode") == target_mode:
                return True
            acts = ep.get("actions", [])
            if len(acts) > 0:
                mode_frac = float(np.mean([int(a) % 5 == target_mode for a in acts]))
                if mode_frac >= 0.8:
                    return True
            return False

        return True

    def _partition_episodes(self) -> tuple[Dict[str, List[dict]], List[dict]]:
        """Partition stored episodes by their unique primary stratum and optional targeted pool."""
        partitions: dict[str, list[dict]] = {
            "agile": [],
            "sparse": [],
            "dense": [],
            "mixed": [],
        }
        targeted_eps: list[dict] = []

        for ep in self.buffer._episodes:
            if self._is_targeted_episode(ep):
                targeted_eps.append(ep)
                continue

            scen_id = ep.get("scenario_id", "unknown")
            scen_cls = ep.get("scenario_class")
            stratum = map_scenario_to_primary_stratum(scen_id, scen_cls)
            # Store primary_stratum in episode dict for auditability
            ep["primary_stratum"] = stratum
            partitions[stratum].append(ep)

        return partitions, targeted_eps

    def _allocate_counts(self, batch_size: int) -> Dict[str, int]:
        """Compute target sequence count per stratum."""
        total_weight = sum(self.strata_weights.values())
        counts: dict[str, int] = {}
        allocated = 0
        keys = list(self.strata_weights.keys())

        for k in keys[:-1]:
            c = int(round(batch_size * (self.strata_weights[k] / total_weight)))
            counts[k] = c
            allocated += c

        # Remainder allocated to last stratum
        counts[keys[-1]] = max(0, batch_size - allocated)
        return counts

    def get_stratum_counts(self) -> Dict[str, int]:
        """Return the count of episodes in each stratum and targeted pool."""
        partitions, targeted_eps = self._partition_episodes()
        counts = {k: len(v) for k, v in partitions.items()}
        if self.targeted_spec:
            counts["targeted"] = len(targeted_eps)
        return counts

    def can_stratify(self, batch_size: int) -> bool:
        """Check if all strata satisfy the minimum episode and transition quotas."""
        if not self.buffer.can_sample(batch_size):
            return False

        partitions, targeted_eps = self._partition_episodes()

        if self.targeted_spec:
            target_frac = float(self.targeted_spec.get("target_fraction", self.targeted_spec.get("fraction", 0.15)))
            n_targeted = int(round(batch_size * target_frac))
            min_targeted = int(self.targeted_spec.get("min_episodes", 2))
            if len(targeted_eps) < min_targeted:
                return False
            tot_target_trans = sum(int(e["length"]) for e in targeted_eps)
            if tot_target_trans < n_targeted:
                return False
            n_strata = batch_size - n_targeted
        else:
            n_strata = batch_size

        counts = self._allocate_counts(n_strata)

        for stratum, needed in counts.items():
            if needed <= 0:
                continue
            eps = partitions[stratum]
            if len(eps) < self.min_episodes_per_stratum:
                return False
            total_trans = sum(int(e["length"]) for e in eps)
            if total_trans < needed:
                return False

        return True

    def sample(
        self,
        batch_size: int,
        target_hit_seq_fraction: float = 0.40,
    ) -> Dict[str, Any]:
        """Sample a batch of sequences with 4-stratum balance and optional targeted quota.

        FAILS CLOSED: If any stratum does not have at least min_episodes_per_stratum,
        or targeted quota cannot be satisfied, raises RuntimeError immediately.
        """
        partitions, targeted_eps = self._partition_episodes()

        if self.targeted_spec:
            target_frac = float(self.targeted_spec.get("target_fraction", self.targeted_spec.get("fraction", 0.15)))
            n_targeted = int(round(batch_size * target_frac))
            min_targeted = int(self.targeted_spec.get("min_episodes", 2))
            if len(targeted_eps) < min_targeted:
                raise RuntimeError(
                    f"G8.3 Fail-Closed Replay Violation: Targeted pool has only {len(targeted_eps)} "
                    f"episodes (minimum {min_targeted} required). "
                    f"Current buffer counts: {self.get_stratum_counts()}"
                )
            tot_target_trans = sum(int(e["length"]) for e in targeted_eps)
            if tot_target_trans < n_targeted:
                raise RuntimeError(
                    f"G8.3 Fail-Closed Replay Violation: Targeted pool has only {tot_target_trans} "
                    f"transitions (needed at least {n_targeted} for batch size {batch_size})."
                )
            n_strata = batch_size - n_targeted
        else:
            n_targeted = 0
            n_strata = batch_size

        counts = self._allocate_counts(n_strata)

        # G8.1 FAIL-CLOSED INVARIANT CHECK
        for stratum, needed in counts.items():
            if needed > 0:
                n_avail = len(partitions[stratum])
                if n_avail < self.min_episodes_per_stratum:
                    raise RuntimeError(
                        f"G8.1 Fail-Closed Replay Violation: Stratum '{stratum}' has only {n_avail} "
                        f"episodes (minimum {self.min_episodes_per_stratum} required). "
                        f"Current buffer counts: {self.get_stratum_counts()}"
                    )
                total_trans = sum(int(e["length"]) for e in partitions[stratum])
                if total_trans < needed:
                    raise RuntimeError(
                        f"G8.1 Fail-Closed Replay Violation: Stratum '{stratum}' has only {total_trans} "
                        f"transitions (needed at least {needed} for batch size {batch_size})."
                    )

        orig_episodes = self.buffer._episodes
        orig_total = self.buffer._total
        orig_current = self.buffer._current
        orig_current_len = self.buffer._current_len

        sub_batches: list[dict[str, Any]] = []

        try:
            # Mask current unarchived episode during stratified sampling
            self.buffer._current = None
            self.buffer._current_len = 0

            # 1. Sample targeted quota if configured
            if n_targeted > 0:
                self.buffer._episodes = targeted_eps
                self.buffer._total = sum(int(e["length"]) for e in targeted_eps)
                sub_batch_targeted = self.buffer.sample(
                    batch_size=n_targeted,
                    target_hit_seq_fraction=target_hit_seq_fraction,
                )
                sub_batches.append(sub_batch_targeted)

            # 2. Sample 4 strata
            for stratum, needed in counts.items():
                if needed <= 0:
                    continue
                stratum_eps = partitions[stratum]
                self.buffer._episodes = stratum_eps
                self.buffer._total = sum(int(e["length"]) for e in stratum_eps)

                sub_batch = self.buffer.sample(
                    batch_size=needed,
                    target_hit_seq_fraction=target_hit_seq_fraction,
                )
                sub_batches.append(sub_batch)
        finally:
            self.buffer._episodes = orig_episodes
            self.buffer._total = orig_total
            self.buffer._current = orig_current
            self.buffer._current_len = orig_current_len

        if not sub_batches:
            raise RuntimeError("G8.1 Fail-Closed: No sub-batches generated during stratified sampling.")

        combined = sub_batches[0]
        for sb in sub_batches[1:]:
            combined = SequenceReplayBuffer.combine_batches(combined, sb)

        return combined
