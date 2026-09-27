import json

arms = ['a', 'b', 'c']
for a in arms:
    with open(f'reports/g8_2r2_eval_arm_{a}_step_26250.json') as f:
        d26 = json.load(f)
    with open(f'reports/g8_2r2_eval_arm_{a}_step_27000.json') as f:
        d27 = json.load(f)
    print(f"=== ARM {a.upper()}2 TRAJECTORY (26,250 -> 27,000) ===")
    print(f"  Mean Pd:       {d26['mean_pd']:.2f}% -> {d27['mean_pd']:.2f}%")
    print(f"  Agile Pd:      {d26['agile_pd']:.2f}% -> {d27['agile_pd']:.2f}%")
    print(f"  Sparse Pd:     {d26['sparse_pd']:.2f}% -> {d27['sparse_pd']:.2f}%")
    print(f"  Dense Pd:      {d26['dense_pd']:.2f}% -> {d27['dense_pd']:.2f}%")
    print(f"  Blackouts:     {d26['blackout_count']} -> {d27['blackout_count']}")
    print(f"  Hmode:         {d26['h_mode']:.3f} -> {d27['h_mode']:.3f}")
    print(f"  Htau:          {d26.get('h_tau', 0.0):.3f} -> {d27.get('h_tau', 0.0):.3f}")
    print(f"  SHORT %:       {d26['short_pct']:.1f}% -> {d27['short_pct']:.1f}%")
    print(f"  NORMAL %:      {d26['normal_pct']:.1f}% -> {d27['normal_pct']:.1f}%")
    print(f"  LONG %:        {d26['long_pct']:.1f}% -> {d27['long_pct']:.1f}%")
    print(f"  REVISIT %:     {d26.get('revisit_pct', 0.0):.1f}% -> {d27.get('revisit_pct', 0.0):.1f}%")
    print(f"  PREEMPTIVE %:  {d26.get('preemptive_pct', 0.0):.1f}% -> {d27.get('preemptive_pct', 0.0):.1f}%")
    print(f"  Q_max:         {d26['q_max']:.2f} -> {d27['q_max']:.2f}")
    print(f"  Q_std:         {d26['q_std']:.2f} -> {d27['q_std']:.2f}")
    print(f"  FiLM Flip %:   {d26['film_flip_pct']:.2f}% -> {d27['film_flip_pct']:.2f}%")
    print()

print("Scenario breakdown at Step 27,000:")
with open('reports/g8_2r2_eval_arm_a_step_27000.json') as f: dA = json.load(f)
with open('reports/g8_2r2_eval_arm_b_step_27000.json') as f: dB = json.load(f)
with open('reports/g8_2r2_eval_arm_c_step_27000.json') as f: dC = json.load(f)

for s in dA['scenarios']:
    pA = dA['scenarios'][s]['pd']
    pB = dB['scenarios'][s]['pd']
    pC = dC['scenarios'][s]['pd']
    print(f"{s:12s} | A2 (27k): {pA:6.2f}% | B2 (27k): {pB:6.2f}% | C2 (27k): {pC:6.2f}%")
