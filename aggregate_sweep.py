"""Combine the noise-sweep JSON files into summary tables."""
import json
from pathlib import Path
import numpy as np

R = Path('results')
SCALES = [0, 0.25, 0.5, 1]
METHODS = ('FRG', 'learned_only', 'checked_learned', 'checked_during_replanning')
GATES = ('t0_learned_gate', 'handoff_scvx_gate')
OUTCOMES = ('correct_accept', 'correct_reject', 'safety_false_accept', 'fuel_false_accept', 'false_reject')


def load(scale):
    reps = [0] if scale == 0 else range(5)
    runs = []
    for r in reps:
        path = R/f'comparison_fourway_n200_s{scale:g}_r{r}.json'
        d = json.load(open(path))
        if d['completed_cases'] != 200:
            raise SystemExit(f'{path} only completed {d["completed_cases"]} cases')
        runs.append(d)
    return runs


def spread(values):
    v = np.asarray(values, float)
    return f'{v.mean():6.1f} [{v.min():g}-{v.max():g}]' if len(v) > 1 else f'{v[0]:6.1f}'


data = {s: load(s) for s in SCALES}

print('\n=== Feasible cases out of 200 (mean over repeats [min-max]) ===')
print(f'{"scale":>6} ' + ' '.join(f'{m:>26}' for m in METHODS))
for s, runs in data.items():
    print(f'{s:>6g} ' + ' '.join(f'{spread([d["methods"][m]["feasible"] for d in runs]):>26}' for m in METHODS))

print('\n=== Passive-safety violations out of 200 ===')
for s, runs in data.items():
    print(f'{s:>6g} ' + ' '.join(f'{spread([d["methods"][m]["pas_violations"] for d in runs]):>26}' for m in METHODS))

print('\n=== Mean fuel saving vs FRG on jointly feasible cases (m/s) [jointly feasible cases] ===')
for s, runs in data.items():
    cells = []
    for m in METHODS[1:]:
        p = [d['pairwise_feasible_comparison'][m] for d in runs]
        saving = [x['mean_fuel_saving_vs_FRG_m_s'] for x in p if x['mean_fuel_saving_vs_FRG_m_s'] is not None]
        n = [x['jointly_feasible_cases'] for x in p]
        cells.append(f'{np.mean(saving) if saving else float("nan"):6.2f} [{np.mean(n):.0f}]')
    print(f'{s:>6g} ' + ' '.join(f'{c:>26}' for c in cells))

for g in GATES:
    print(f'\n=== {g}: outcomes summed over repeats ===')
    print(f'{"scale":>6} {"decisions":>9} ' + ' '.join(f'{o:>20}' for o in OUTCOMES) + f' {"unsafe/accepts":>15}')
    for s, runs in data.items():
        tot = {o: sum(d['gate_outcomes'][g][o] for d in runs) for o in OUTCOMES}
        accepts = tot['correct_accept'] + tot['safety_false_accept'] + tot['fuel_false_accept']
        rate = tot['safety_false_accept']/accepts if accepts else float('nan')
        print(f'{s:>6g} {200*len(runs):>9} ' + ' '.join(f'{tot[o]:>20}' for o in OUTCOMES) + f' {rate:>15.3f}')

print('\n=== Gate fuel prediction error: true minus predicted (m/s), all cases ===')
print(f'{"scale":>6} {"learned mean":>13} {"learned >0":>11} {"FRG mean":>10} {"FRG >0":>8}')
for s, runs in data.items():
    lg, fg = [], []
    for d in runs:
        for case in d['cases'].values():
            lg.append(case['learned_only']['fuel_m_s']-case['learned_only']['predicted_fuel_m_s'])
            fg.append(case['FRG']['fuel_m_s']-case['FRG']['predicted_fuel_m_s'])
    lg, fg = np.array(lg), np.array(fg)
    print(f'{s:>6g} {lg.mean():13.2f} {np.mean(lg > 1e-6):11.2f} {fg.mean():10.2f} {np.mean(fg > 1e-6):8.2f}')

# import json, numpy as np
# d = json.load(open('results/comparison_fourway_n200_s0_r0.json'))['cases']
# pas = [v['checked_during_replanning']['pas_m'] for v in d.values()
#        if v['checked_during_replanning']['switched_to_replanner']]
# pas = np.array(pas)
# print(len(pas), 'switched cases; passive-separation margin above 500 m:')
# print(np.round(np.sort(pas - 500), 1))