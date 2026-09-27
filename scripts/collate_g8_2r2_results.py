import json
from pathlib import Path

reports = {
    'Arm A2 (c=2.0, tau_ref=1.0)': 'reports/g8_2r2_eval_arm_a_step_26250.json',
    'Arm B2 (c=0.0)': 'reports/g8_2r2_eval_arm_b_step_26250.json',
    'Arm C2 (c=1.0, tau_ref=1.0)': 'reports/g8_2r2_eval_arm_c_step_26250.json',
}

data = {}
for name, p in reports.items():
    with open(p) as f:
        data[name] = json.load(f)

print(f"| {'Metric':30s} | {'Arm A2 (Control)':18s} | {'Arm B2 (No Imm Dwell)':22s} | {'Arm C2 (Centered)':20s} |")
print("|:" + "-"*30 + "-|:" + "-"*18 + "-|:" + "-"*22 + "-|:" + "-"*20 + "-|")
metrics = [
    ('Mean Pd (%)', 'mean_pd', ':.2f'),
    ('Agile Pd (%)', 'agile_pd', ':.2f'),
    ('Sparse Pd (%)', 'sparse_pd', ':.2f'),
    ('Dense Pd (%)', 'dense_pd', ':.2f'),
    ('Blackout Count', 'blackout_count', 'd'),
    ('H_mode (Entropy)', 'h_mode', ':.3f'),
    ('SHORT Dwell (%)', 'short_pct', ':.2f'),
    ('NORMAL Dwell (%)', 'normal_pct', ':.2f'),
    ('LONG Dwell (%)', 'long_pct', ':.2f'),
    ('Q_max', 'q_max', ':.2f'),
    ('Q_mean', 'q_mean', ':.2f'),
    ('Q_std', 'q_std', ':.2f'),
    ('Greedy Action Margin', 'greedy_action_margin', ':.4f'),
    ('FiLM Flip Rate (%)', 'film_flip_pct', ':.2f'),
    ('FiLM Rho', 'film_rho', ':.6f'),
]

for label, key, fmt in metrics:
    vA = format(data['Arm A2 (c=2.0, tau_ref=1.0)'][key], fmt[1:])
    vB = format(data['Arm B2 (c=0.0)'][key], fmt[1:])
    vC = format(data['Arm C2 (c=1.0, tau_ref=1.0)'][key], fmt[1:])
    print(f"| {label:30s} | {vA:18s} | {vB:22s} | {vC:20s} |")

print("\nScenario Breakdown:")
scens = list(data['Arm A2 (c=2.0, tau_ref=1.0)']['scenarios'].keys())
for s in scens:
    pA = data['Arm A2 (c=2.0, tau_ref=1.0)']['scenarios'][s]['pd']
    pB = data['Arm B2 (c=0.0)']['scenarios'][s]['pd']
    pC = data['Arm C2 (c=1.0, tau_ref=1.0)']['scenarios'][s]['pd']
    print(f"{s:12s} | A2: {pA:6.2f}% | B2: {pB:6.2f}% | C2: {pC:6.2f}%")
