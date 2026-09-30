# Far-range rendezvous recovery

Minimal code and saved inputs for three approaches in a far-range, angles-only rendezvous setting: fixed-reference guidance (FRG), learned reference replacement (DLR/VLRR), and provisional guidance during replanning (PGDR). FRG is an independent reconstruction from the equations of [Wijayatunga, Armellin, and Holt (2025)](https://doi.org/10.1016/j.ast.2025.109996), not their published implementation or trained policy.

| File | Purpose |
| --- | --- |
| `minduli_setting.py` | Mission dynamics, nominal path, PSO, and generation of 1,000 disturbed-state recovery cases. |
| `minduli_compare.py` | FRG controller; learned replacement planner; unscreened and screened evaluation; optional training. |
| `minduli_interim.py` | PGDR: apply the learned first burn, then run PSO from the state reached after a modeled planning delay. |
| `data/minduli_setting_data.npz` | The 1,000 recovery cases. |
| `results/fixed_ppo.pt`, `results/latent_ppo.pt` | Saved weights for the two policies. |

The cases comprise ten perturbations at each of 100 nominal points. The split by nominal point is 700 training, 100 validation, and 200 held-out cases.

## Run

Python 3.9 with the versions in `requirements.txt` was used for the saved runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python minduli_setting.py self-check
python minduli_setting.py audit
python minduli_compare.py --self-check
python minduli_interim.py --self-check
python minduli_compare.py
python minduli_interim.py
```

`minduli_compare.py` evaluates the saved weights on the 200 held-out cases and writes `results/comparison.json`. `minduli_interim.py` runs the 200-case PGDR comparison, writes `results/interim_same_pipeline.json`, and takes longer because it solves PSO at each handoff. To regenerate the data and retrain the policies, run `python minduli_setting.py generate` followed by `python minduli_compare.py --train`. Retraining is stochastic and need not reproduce the saved weights exactly.

## Scope

The tests supply the disturbed state directly and assume ideal thrust. Angles-only navigation is assessed by filter covariance rather than a closed-loop estimator. VLRR uses stored PSO plans as fallback proxies. PGDR models a 108 ms planning delay and runs PSO after the provisional burn, not concurrently with it. The finite-budget PSO solutions are not global optima. Report fuel together with terminal error, point-wise and passive safety, burn limits, and navigation uncertainty.
