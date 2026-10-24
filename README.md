# Far-range rendezvous recovery

Code and saved inputs for comparing four recovery strategies in a far-range, angles-only rendezvous setting. FRG is an independent reconstruction from the equations of [Wijayatunga, Armellin, and Holt (2025)](https://doi.org/10.1016/j.ast.2025.109996), not their published implementation or trained policy.

| File | Purpose |
| --- | --- |
| `minduli_setting.py` | Mission dynamics, nominal path, PSO, and generation of 1,000 disturbed-state recovery cases. |
| `minduli_compare.py` | Main four-way experiment entry point; policy definitions, FRG, covariance evaluation, and optional training. |
| `minduli_fourway.py` | Common execution/validation, checked-learned selection, SCvx handoff, and matched-case summaries. |
| `minduli_interim.py` | PGDR: apply the learned first burn, then run PSO from the state reached after a modeled planning delay. |
| `minduli_scvx.py` | Standalone SCvx-style constrained recovery planner, with cold, stored-PSO, or learned initialization. |
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
```

`minduli_compare.py` now evaluates the saved weights on the 200 held-out cases and writes `results/comparison_fourway.json`. The old `comparison.json` is a historical experiment with a stored-PSO fallback, not the current four-way result. `minduli_interim.py` is also retained as a separate historical PSO-delay experiment; it is not called by the main comparison. To regenerate the data and retrain the policies, run `python minduli_setting.py generate` followed by `python minduli_compare.py --train`. Retraining is stochastic and need not reproduce the saved weights exactly.

## Four-way comparison setup

| Method | After the disturbance |
| --- | --- |
| FRG | Correct toward the original reference through arrival. |
| Learned only | Track one learned recovery reference through arrival with the same alpha/QCQP guidance, without screening. |
| Checked learned | Track a feasible learned reference only if it improves on feasible nominal guidance; otherwise track the original reference. A feasible recovery is allowed when nominal guidance is infeasible. |
| Checked learned during replanning | Follow the accepted learned plan, otherwise FRG, during the modeled planning delay. Switch to a feasible SCvx continuation only if it improves on both feasible nominal and incumbent continuations. Keep the same alpha/QCQP guidance; retain the bridge reference on rejection or a missed deadline. |

```bash
python minduli_compare.py --self-check
python minduli_compare.py --cases 5 --planning-delay-s 60 --replanner-iterations 30
python minduli_compare.py --planning-delay-s 60 --replanner-iterations 30
```

The default is all 200 held-out states, with no retraining, true-state inputs,
ideal thrust, and nonlinear two-body propagation. Learned and SCvx recovery
references each have six newly spaced maneuver nodes. All methods use the
same saved alpha-selection policy and QCQP correction law on the original
540 s guidance clock, augmented by reference burn nodes and handoff. Feedforward
commands come from the selected reference. Policy history resets at disturbance
and reference handoff; weights never change. The saved alpha policy was trained
on the original reference, so its transfer to replacement references is evaluated
without retraining. With ideal propagation and a replacement reference starting
at the exact disturbed state, tracking corrections can be nearly zero. This run
isolates reference selection; it does not establish robustness to later navigation
or thrust errors. The target is unchanged: a 1 km along-track relative state at
the original six-hour arrival. Every executed trajectory is assessed using the
same terminal tolerances (1 m / 0.01 m/s), 500 m point-wise and missed-burn
passive safety, and 10 m/s ECI command-component bounds. Final position sigma
must not exceed nominal from the same disturbance phase. Covariance uses a
common 10 s measurement cadence, independent of burn or handoff count.

Reference acceptance prioritizes feasibility, then lower predicted fuel. At the
disturbance, the checked variants compare the guided learned-reference rollout
against original-reference guidance from the same state and time. When nominal
guidance is feasible, the candidate must save more than 1e-6 m/s. Ties and more
expensive feasible candidates are rejected. If nominal guidance is infeasible,
a feasible recovery may be accepted even when its fuel is higher; this is logged
as `feasible_recovery`, not a fuel improvement. Learned-only remains an unscreened
control baseline, with its comparison recorded but not enforced.

At handoff, SCvx explicitly minimizes fuel. Its guided candidate is compared with
both continuing the incumbent plan and returning to original-reference guidance
from the same predicted handoff state and time. All alternatives share the
executed prefix, its spent fuel, and its covariance history. A switch requires
improvement against each feasible baseline and full-trajectory feasibility. An
already incurred safety violation cannot be repaired retroactively. Total fuel
in the results includes the bridge; recorded remaining-fuel comparisons exclude
that common prefix. Per-case records include comparison costs, feasibility,
savings, acceptance/rejection reasons, and deadline checks. This selection rule
prevents a predicted fuel increase over a feasible incumbent; it does not certify
a global optimum or improvement under unmodeled execution errors.

The **60 s delay is an assumed simulation budget**, not a measured PSO or
flight-processor runtime. It is configurable with `--planning-delay-s`. SCvx
plans from the predicted state at handoff while the bridge executes in the
simulation. Ideal dynamics make that prediction exact. Measured desktop SCvx
wall time is recorded separately, and a plan that takes longer than the modeled
budget is not used. Candidate-generation/checking times are measured but are
not inserted into the simulated burn schedule. This is a deterministic latency
experiment, not flight-hardware or closed-loop estimator validation.

Each JSON case records all four executed outcomes, candidate acceptance,
SCvx termination/convergence, switch decisions, and solver timing. Aggregates
separate physical feasibility from the additional covariance check. Fuel is
reported both over all cases (including failures) and on jointly feasible
cases; only the latter supports a matched efficiency comparison. Partial
results are saved after every case. `--cases` evenly samples the test split.

## Four-way results

The earlier direct-execution comparison is archived in
`results/comparison_fourway_direct_execution_v1.json`. It used different
execution mechanisms for FRG and replacement plans, so it does not isolate
reference selection. Its table and illustration must not be used for the
corrected shared-guidance comparison.

The shared-guidance, feasibility-only experiment is archived in
`results/comparison_fourway_shared_guidance_v2.json`. Its results predate the
fuel-improvement acceptance rule.

The current protocol is `shared-guidance-fuel-improvement-v3`. Results are written
to `results/comparison_fourway.json`; `completed_cases` distinguishes partial runs
from the complete 200-case evaluation.

The updated run completed all 200 held-out cases with the saved policies, a
60 s assumed handoff budget, and at most 30 SCvx iterations.

| Method | Physical feasibility (higher is better) | Including covariance (higher is better) | Mean fuel on the same 15 cases feasible for all four (m/s; lower is better) |
| --- | ---: | ---: | ---: |
| FRG | 75/200 | 33/200 | 16.80 |
| Learned only (unscreened baseline) | 83/200 | 81/200 | 17.34 |
| Checked learned | 131/200 | 99/200 | 13.94 |
| Checked learned during replanning | 141/200 | 126/200 | 11.84 |

The learned checker accepted 74 references: 8 fuel improvements and 66 feasible
recoveries over infeasible nominal guidance. Seven otherwise feasible learned
references were rejected for lack of a fuel improvement. SCvx supplied 68
accepted reference switches. Every selected replacement passed the full checker;
all 200 cases passed the per-case non-increasing-fuel audit whenever the
comparison baseline was feasible. On the 33 cases feasible for FRG and each
checked alternative, average savings versus FRG were 1.30 m/s for checked learned
and 2.54 m/s for checked learned during replanning.

The replanning feasibility count is 126/200, versus 147/200 in the prior
feasibility-only, mission-objective run. The new fuel objective and stricter
selection trade some recovery coverage for fuel improvement. These are predicted
ideal-model results, not a global-optimality or flight-robustness guarantee.

## SCvx recovery

Install the updated requirements, then run:

```bash
python minduli_scvx.py --self-check
python minduli_scvx.py --case 0 --initialization cold
python minduli_scvx.py --case 0 --initialization learned
python minduli_scvx.py --case 0 --initialization pso
```

`--case` selects an index in the saved dataset (not all indices are held-out).
Each run writes `results/scvx_<case>_<initialization>_<objective>.json`, including
the plan, constraint checks, trust-region history, termination status, and timing.
Cold initialization starts the first four burns at zero. The CLI exits with code
2 after saving diagnostics if the returned plan fails validation.
The stored-PSO initialization reads an already generated plan; its timing excludes
the offline PSO solve. Learned initialization timing includes checkpoint loading.
These single-case runs are diagnostics, not a matched runtime benchmark.

The default `--objective mission` retains `score()`'s normalized fuel,
line-of-sight geometry, and Gaussian safety costs, while handling clearance and
burn limits with explicit linearized constraints. `--objective fuel` instead
minimizes total impulse magnitude and is a different optimization problem.
Both use fixed six-burn times, nonlinear two-body propagation, and terminal
shooting for the final two burns. `warm` in the callable `scvx()` accepts either
a `(4, 3)` or `(6, 3)` array of ECI burns.

This is a reduced-space SCvx-style implementation: central finite differences,
elastic constraint slacks, and a merit-ratio trust region solved with CVXPY and
Clarabel. It does not inherit a published SCvx convergence or feasibility theorem.
The returned `feasible` flag requires independent nonlinear validation of terminal
errors, every ECI burn component, and refined point-wise/passive closest approaches.
The validator brackets minima on a 10 s grid and refines negative-to-positive
range-rate crossings on each unforced coast. This catches violations missed by
fixed-grid checks, but is still numerical validation, not a formal certificate.
Slacks alone never authorize a plan. An iteration limit can still
return the best validated feasible iterate; `converged` is reported separately.
Always check `feasible` before using a returned plan, and handle failure explicitly.

`--max-sigma METRES` adds the existing angles-only covariance calculation as a
final screen. Without it, covariance is not checked; the geometry cost is not a
covariance guarantee. Even with it, covariance is screened rather than constrained
inside the convex subproblems. Numerical safety checks do not certify continuous-time
safety or closed-loop robustness. The main four-way experiment uses this solver
for the provisional-guidance handoff with `objective="fuel"`. The separate historical interim script
still uses PSO, and PSO data generation and policy training are unchanged.

Previously saved safety metrics have not been regenerated after the validator
change. In particular, saved PSO case 153 actually enters the 500 m KOZ; do not
use its stored feasibility label as a validated fallback certificate.

## Scope

The tests supply the disturbed state directly and assume ideal thrust. Angles-only navigation is assessed by filter covariance rather than a closed-loop estimator. The main four-way comparison uses FRG fallback and an assumed SCvx arrival budget. The historical interim script still models a 108 ms PSO delay and solves PSO after the provisional burn. Neither finite-budget PSO nor the SCvx-style implementation certifies a global optimum. Report fuel together with terminal error, point-wise and passive safety, burn limits, and navigation uncertainty.
