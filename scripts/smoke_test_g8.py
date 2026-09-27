"""Smoke test script for Phase G8 modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

print("=== 1. Testing Imports ===")
from ew_core.models.film_factorized_drqn import FiLMGatedFactorizedDRQN
from ew_core.training.reward_g8 import RelativeDwellShaper, cosine_beta_schedule
from ew_core.training.mode_collapse_guard import ModeCollapseGuard
from ew_core.training.stratified_replay_sampler import StratifiedReplaySampler, map_scenario_class_to_stratum
from ew_core.training.replay_buffer import SequenceReplayBuffer
print("Imports successful!")

print("=== 2. Testing G7-A Checkpoint Loading ===")
g7a_path = Path("experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_26000.pt")
assert g7a_path.exists(), f"Missing {g7a_path}"
model, manifest = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(g7a_path, seed=42)
print(f"Inherited params count: {manifest['inherited_params_count']}")
print(f"Dropped keys: {manifest['dropped_keys']}")
assert manifest['inherited_params_count'] == 34, f"Expected 34 inherited param tensors, got {manifest['inherited_params_count']}"
assert set(manifest['dropped_keys']).issubset({"interaction_proj.weight", "interaction_mode.weight"}), f"Unexpected dropped keys: {manifest['dropped_keys']}"

g7b_path = Path("experiments/checkpoints/g7b_low_rank_coupling/checkpoint_gate_26000.pt")
if g7b_path.exists():
    model_b, manifest_b = FiLMGatedFactorizedDRQN.from_g7a_checkpoint(g7b_path, seed=42)
    print(f"G7-B Dropped keys: {manifest_b['dropped_keys']}")
    assert set(manifest_b['dropped_keys']) == {"interaction_proj.weight", "interaction_mode.weight"}

print("=== 3. Testing FiLM Parameter Count and Identity Init ===")
film_params = sum(p.numel() for p in model.film_proj.parameters())
print(f"FiLM parameters: {film_params}")
assert film_params == 2410, f"Expected 2410 FiLM params, got {film_params}"
assert torch.all(model.film_proj[2].weight == 0.0), "film_proj[2].weight is not zero!"
assert torch.all(model.film_proj[2].bias == 0.0), "film_proj[2].bias is not zero!"

print("=== 4. Testing Zero-Step Parity against Pure Factorized Q ===")
B, T, D = 2, 4, 360
obs = torch.randn(B, T, D)
q_flat, aux, hidden = model(obs)
assert q_flat.shape == (B, T, 180), f"Unexpected q shape: {q_flat.shape}"
assert "intercept_prob" in aux and aux["intercept_prob"].shape == (B, T, 180)
assert "intercept_time_us" in aux and aux["intercept_time_us"].shape == (B, T, 180)

# Check pure factorized parity:
# Q_expected = V(s) + A_band_tilde + A_mode_tilde (since gamma=1.0 and beta=0.0)
v = aux["v"].unsqueeze(-1)  # (B, T, 1, 1)
a_b = aux["a_band_tilde"].unsqueeze(-1)  # (B, T, 36, 1)
a_m = aux["a_mode_tilde"].unsqueeze(2)   # (B, T, 1, 5) -> modulated with gamma=1, beta=0
q_pure_factorized = (v + a_b + a_m).view(B, T, 180)
max_diff = torch.max(torch.abs(q_flat - q_pure_factorized)).item()
print(f"Max diff between FiLM output and pure factorized: {max_diff:.8e}")
assert max_diff < 1e-6, f"Zero-step parity violated! max_diff={max_diff}"

print("=== 5. Testing act() Method ===")
single_obs = torch.randn(360)
action, next_h = model.act(single_obs)
assert isinstance(action, int) and 0 <= action < 180
assert "raw_drqn_action" in model.last_decision_telemetry
assert model.last_decision_telemetry["decision_source"] == "ml_exploitation"
print(f"act() returned action={action}, telemetry={model.last_decision_telemetry['final_action']}")

print("=== 6. Testing RelativeDwellShaper ===")
shaper = RelativeDwellShaper(ema_alpha=0.05, c_dwell=2.0)
r1 = shaper.shape(raw_reward=10.0, mode_chosen=0, dwell_us=125.0)  # SHORT
r2 = shaper.shape(raw_reward=10.0, mode_chosen=2, dwell_us=1250.0) # LONG
print(f"Shaped reward SHORT: {r1:.4f}, LONG: {r2:.4f}")
assert np.isfinite(r1) and np.isfinite(r2)

print("=== 7. Testing cosine_beta_schedule ===")
b_start = cosine_beta_schedule(26000, 27000, start_step=26000, beta_max=0.10, warmup_frac=0.10, beta_min=0.01)
b_mid = cosine_beta_schedule(26100, 27000, start_step=26000, beta_max=0.10, warmup_frac=0.10, beta_min=0.01)
b_end = cosine_beta_schedule(27000, 27000, start_step=26000, beta_max=0.10, warmup_frac=0.10, beta_min=0.01)
print(f"Beta at 26000: {b_start:.4f}, at 26100 (peak): {b_mid:.4f}, at 27000 (end): {b_end:.4f}")
assert abs(b_start - 0.01) < 1e-4, f"Expected 0.01 at start, got {b_start}"
assert abs(b_mid - 0.10) < 1e-4, f"Expected 0.10 at warmup peak, got {b_mid}"
assert abs(b_end - 0.01) < 1e-4, f"Expected 0.01 at end, got {b_end}"

print("=== 8. Testing ModeCollapseGuard ===")
guard = ModeCollapseGuard(window=500, short_ceiling=0.80, min_entropy=0.30, min_samples=100)
# Feed 150 mixed modes
for i in range(150):
    res = guard.update(i % 5)
assert not res["triggered"], "Spurious trigger on mixed modes!"
print(f"Mixed modes: short_frac={res['short_frac']:.2f}, entropy={res['entropy']:.3f}, triggered={res['triggered']}")
# Now feed 400 SHORT modes (mode 0)
for _ in range(400):
    res = guard.update(0)
print(f"Runaway SHORT: short_frac={res['short_frac']:.2f}, entropy={res['entropy']:.3f}, triggered={res['triggered']}")
assert res["triggered"], "ModeCollapseGuard failed to trip on runaway SHORT!"

print("=== 9. Testing StratifiedReplaySampler ===")
buf = SequenceReplayBuffer(capacity=5000, seq_len=16, obs_dim=360, burn_in=8, seed=42)
scen_classes = ["fast_agile", "sparse", "dense", "mixed"]
for sc_idx, sc_cls in enumerate(scen_classes):
    for step_i in range(50):
        buf.add(
            obs=np.random.randn(360).astype(np.float32),
            action=int(step_i % 180),
            reward=1.0 if step_i % 5 == 0 else 0.0,
            next_obs=np.random.randn(360).astype(np.float32),
            done=bool(step_i == 49),
            hit_prob=1.0 if step_i % 5 == 0 else 0.0,
            scenario_id=f"scen_{sc_cls}_{sc_idx}",
            dwell_time_us=500.0,
        )
assert buf.n_episodes() == 4, f"Expected 4 episodes, got {buf.n_episodes()}"
sampler = StratifiedReplaySampler(buf, min_episodes_per_stratum=1)
assert sampler.can_stratify(batch_size=16)
batch = sampler.sample(batch_size=16)
assert batch["obs"].shape == (16, 16, 360)
print(f"Stratified batch sampled successfully: shape={batch['obs'].shape}")

print("=== 10. Testing FiLM Gradient Flow ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = q_flat.sum()
loss.backward()
grad_norm_film0 = model.film_proj[0].weight.grad.norm().item()
grad_norm_film2 = model.film_proj[2].weight.grad.norm().item()
print(f"Gradient norm film_proj[0]: {grad_norm_film0:.6f}, film_proj[2]: {grad_norm_film2:.6f}")
assert grad_norm_film2 > 0.0, "film_proj[2] did not receive gradient!"

print("\nALL 10 SMOKE TESTS PASSED CLEANLY!")
