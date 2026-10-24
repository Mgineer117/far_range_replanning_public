"""Reduced-space successive convexification for the six-impulse RPO model.

Nonlinear shooting eliminates the states and terminal burn pair. Convex steps
linearize the remaining burn and sampled clearance maps, with elastic slacks
and a merit-ratio trust region. This is an SCvx-style numerical baseline, not
an implementation carrying the convergence theorem of a particular SCvx paper.
"""

import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

import minduli_setting as m

ROOT = Path(__file__).resolve().parent


def rollout(start, times, candidates):
    """Batch exact two-body rollouts on the existing validation sample grids."""
    burns, miss = m.complete_last_two(start, times, candidates.reshape(-1, 4, 3))
    x = np.broadcast_to(m.to_eci(start, times[0]), (len(burns), 6)).copy()
    grid = np.linspace(times[0], times[-1], max(1000, int((times[-1]-times[0])/10)))
    clearances = []
    for j, t in enumerate(times):
        # Passive safety is checked before each commanded burn, including final.
        probe = t + np.linspace(0, 5400, 120)
        free = m.to_rel(m.propagate(x[:, None], probe[None]-t), probe[None])
        clearances.append(np.linalg.norm(free[..., :3], axis=-1))
        x[:, 3:] += burns[:, j]
        if j < 5:
            samples = grid[(grid >= t-1e-8) & (grid < times[j+1]-1e-8)]
            # Include both segment endpoints as well as the validation grid.
            samples = np.unique(np.r_[t, samples, times[j+1]])
            path = m.to_rel(m.propagate(x[:, None], samples[None]-t), samples[None])
            clearances.append(np.linalg.norm(path[..., :3], axis=-1))
            x = m.propagate(x, times[j+1]-t)
    return burns, np.concatenate(clearances, axis=1), miss


def objectives(start, times, candidates, burns, objective):
    fuel = np.linalg.norm(burns, axis=-1).sum(axis=1)
    if objective == 'fuel':
        return fuel, fuel
    # Recover precisely the existing smooth fuel/geometry/Gaussian safety cost;
    # hard-violation penalties are handled separately by the SCvx merit function.
    value, _, pws, pas = m.score(start, times, candidates.reshape(-1, 4, 3))
    overflow = np.maximum(np.abs(burns[:, 4:])-10, 0).sum(axis=(1, 2))
    violation = np.maximum(m.KOZ-pws, 0)/m.KOZ + np.maximum(m.KOZ-pas, 0)/m.KOZ
    _, miss = m.complete_last_two(start, times, candidates.reshape(-1, 4, 3))
    value -= 100*overflow + 100_000*overflow**2 + 100_000*violation**2 + miss
    return value, fuel/1.7446


def assess(start, times, burns, max_sigma=None):
    pws, pas = m.validated_safety(start, times, burns)
    final = m.trajectory(start, times, burns, np.array([times[-1]]))[0]
    result = {
        'fuel_m_s': float(np.linalg.norm(burns, axis=1).sum()),
        'pws_m': pws, 'pas_m': pas,
        'terminal_position_m': float(np.linalg.norm(final[:3]-m.GOAL[:3])),
        'terminal_velocity_m_s': float(np.linalg.norm(final[3:]-m.GOAL[3:])),
        'max_burn_component_m_s': float(np.abs(burns).max()),
        'covariance_screened': max_sigma is not None,
    }
    if max_sigma is not None:
        import minduli_compare as c
        x = m.to_eci(start, times[0])
        trace = []
        for j in range(5):
            x[3:] += burns[j]
            trace.append((times[j], times[j+1], x.copy()))
            x = m.propagate(x, times[j+1]-times[j])
        result['position_sigma_m'] = c.covariance_sigma(trace)
    result['feasible'] = bool(
        all(np.isfinite(v) for v in result.values()) and
        result['terminal_position_m'] < 1 and
        result['terminal_velocity_m_s'] < .01 and
        min(pws, pas) >= m.KOZ and
        result['max_burn_component_m_s'] <= 10+1e-7 and
        (max_sigma is None or result['position_sigma_m'] <= max_sigma))
    return result


def scvx(start, t0, warm=None, iterations=30, objective='mission', max_sigma=None):
    """Return a plan plus diagnostics; execute only if metrics['feasible'].

    warm is either the first four ECI burns (4,3) or a complete (6,3) plan.
    Covariance, when requested, is a final screen, not a convexified constraint.
    """
    started = time.perf_counter()
    start = np.asarray(start, dtype=float)
    if (start.shape != (6,) or not np.all(np.isfinite(start)) or
            not np.isfinite(t0) or not 0 <= t0 < m.TOF or
            iterations < 1 or objective not in ('fuel', 'mission') or
            (max_sigma is not None and (not np.isfinite(max_sigma) or max_sigma <= 0))):
        raise ValueError('Invalid state, start time, iteration budget, objective, or sigma limit')
    if warm is None:
        z = np.zeros(12)
    else:
        warm = np.asarray(warm, dtype=float)
        if warm.shape not in ((4, 3), (6, 3)) or not np.all(np.isfinite(warm)):
            raise ValueError('warm must be finite with shape (4,3) or (6,3)')
        z = warm[:4].reshape(12).copy()
    times = np.linspace(t0, m.TOF, 6)
    penalty, radius, epsilon = 10_000., 1., 1e-3
    history = []
    status = 'iteration_limit'
    best = None
    initial_burns = m.complete_last_two(start, times, z.reshape(4, 3))[0][0]
    initial_metrics = assess(start, times, initial_burns, max_sigma)
    if initial_metrics['feasible']:
        initial_cost = objectives(start, times, z[None], initial_burns[None], objective)[0][0]
        best = (float(initial_cost), initial_burns.copy(), initial_metrics)

    def evaluate(points):
        burns, distance, miss = rollout(start, times, points)
        values, fuel_term = objectives(start, times, points, burns, objective)
        # A small clearance buffer protects against final nonlinear roundoff.
        g = np.concatenate(((distance-m.KOZ-.1)/m.KOZ,
                            (10-burns.reshape(-1, 18))/10,
                            (10+burns.reshape(-1, 18))/10), axis=1)
        return burns, g, values, fuel_term, miss

    for k in range(iterations):
        points = np.concatenate((z[None], z+epsilon*np.eye(12), z-epsilon*np.eye(12)))
        burns, g, values, fuel_term, miss = evaluate(points)
        if not all(np.all(np.isfinite(a)) for a in (burns, g, values, miss)):
            status = 'nonfinite_rollout'
            break
        b0, g0, f0 = burns[0], g[0], values[0]
        violation = np.maximum(-g0, 0).sum()
        merit = f0 + penalty*violation
        if violation < 1e-8 and miss[0] < 1:
            metrics = assess(start, times, b0, max_sigma)
            if metrics['feasible'] and (best is None or f0 < best[0]):
                best = (f0, b0.copy(), metrics)
        jb = ((burns[1:13]-burns[13:])/ (2*epsilon)).reshape(12, 18).T
        jg = ((g[1:13]-g[13:])/(2*epsilon)).T
        gradient = ((values-fuel_term)[1:13]-(values-fuel_term)[13:])/(2*epsilon)
        step = cp.Variable(12)
        slack = cp.Variable(len(g0), nonneg=True)
        affine_burns = cp.reshape(b0.reshape(18)+jb@step, (6, 3), order='C')
        scale = 1. if objective == 'fuel' else 1/1.7446
        model_cost = (scale*cp.sum(cp.norm(affine_burns, axis=1)) +
                      (f0-fuel_term[0]) + gradient@step)
        problem = cp.Problem(cp.Minimize(model_cost+penalty*cp.sum(slack)),
                             [g0+jg@step+slack >= 0, cp.norm_inf(step) <= radius])
        try:
            problem.solve(solver='CLARABEL')
        except cp.error.SolverError:
            status = 'solver_error'
            break
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or step.value is None:
            status = 'subproblem_'+str(problem.status)
            break
        delta = step.value
        predicted = merit-float(problem.value)
        trial = z+delta
        tb, tg, tf, _, tm = evaluate(trial[None])
        trial_merit = tf[0]+penalty*np.maximum(-tg[0], 0).sum()
        ratio = (merit-trial_merit)/predicted if predicted > 1e-9 else -np.inf
        accepted = bool(np.isfinite(trial_merit) and ratio > .1)
        history.append({'iteration': k+1, 'objective': float(f0),
                        'violation': float(violation), 'radius_m_s': radius,
                        'accepted': accepted,
                        'ratio': float(ratio) if np.isfinite(ratio) else None})
        if accepted:
            z = trial
            if np.maximum(-tg[0], 0).sum() < 1e-8 and tm[0] < 1:
                metrics = assess(start, times, tb[0], max_sigma)
                if metrics['feasible'] and (best is None or tf[0] < best[0]):
                    best = (float(tf[0]), tb[0].copy(), metrics)
            if ratio > .75:
                radius = min(5., 2*radius)
        else:
            radius *= .5
        if violation < 1e-8 and np.linalg.norm(delta, np.inf) < 1e-4:
            status = 'converged'
            break
        if radius < 1e-5:
            status = 'trust_region_limit'
            break
    if best is None:
        burns = m.complete_last_two(start, times, z.reshape(4, 3))[0][0]
        metrics = assess(start, times, burns, max_sigma)
        # Keep validation and algorithm termination distinct, even on failure.
    else:
        _, burns, metrics = best
    cost = objectives(start, times, burns[:4].reshape(1, 12), burns[None], objective)[0][0]
    metrics.update(status=status, converged=status == 'converged',
                   objective_value=float(cost), iterations=len(history), objective=objective,
                   elapsed_s=time.perf_counter()-started, history=history)
    return times, burns, metrics


def self_check(data):
    i = 0
    start, t0 = data['starts'][i], float(data['replan_times'][i, 0])
    warm = data['replan_burns'][i]
    initial = assess(start, data['replan_times'][i], warm)
    assert initial['feasible'], initial
    times, burns, result = scvx(start, t0, warm, iterations=8, objective='fuel')
    assert result['feasible'], result
    assert result['fuel_m_s'] < initial['fuel_m_s']-.01, result
    # Verify safety-map derivatives against an independent directional rollout.
    z = warm[:4].reshape(12)
    direction = np.random.default_rng(1).normal(size=12)
    eps = 1e-3
    _, d, _ = rollout(start, times, np.r_[z[None], z+eps*np.eye(12), z-eps*np.eye(12)])
    jac = ((d[1:13]-d[13:])/(2*eps)).T
    _, moved, _ = rollout(start, times, (z+1e-4*direction)[None])
    linear_change = np.einsum('ij,j->i', jac, 1e-4*direction)
    assert np.max(np.abs(moved[0]-d[0]-linear_change)) < .1
    unsafe = burns.copy()
    unsafe[0, 0] = 11
    assert not assess(start, times, unsafe)['feasible']
    print(json.dumps({'self_check': 'passed', 'initial_fuel_m_s': initial['fuel_m_s'],
                      **{k: v for k, v in result.items() if k != 'history'}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--initialization', choices=('cold', 'pso', 'learned'), default='cold')
    parser.add_argument('--objective', choices=('mission', 'fuel'), default='mission')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--max-sigma', type=float, help='Optional final covariance screen, metres')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    with np.load(ROOT/'data'/'minduli_setting_data.npz') as data:
        if args.self_check:
            self_check(data)
            return
        if not 0 <= args.case < len(data['starts']):
            parser.error('--case is outside the saved dataset')
        warm = None
        initialization_started = time.perf_counter()
        if args.initialization == 'pso':
            warm = data['replan_burns'][args.case]
        elif args.initialization == 'learned':
            import minduli_compare as c
            import minduli_interim as interim
            import torch
            c.NOM_TIMES, c.NOM_BURNS, _ = m.nominal()
            policy = c.ReplanPolicy()
            policy.load_state_dict(torch.load(ROOT/'results'/'latent_ppo.pt',
                                              map_location='cpu', weights_only=True))
            policy.eval()
            warm = interim.predicted_plan(policy, data, args.case)[1]
        initialization_s = time.perf_counter()-initialization_started
        times, burns, metrics = scvx(data['starts'][args.case],
                                     float(data['replan_times'][args.case, 0]), warm,
                                     args.iterations, args.objective, args.max_sigma)
    output = {'case': args.case, 'initialization': args.initialization,
              'initialization_s': initialization_s, 'times': times.tolist(),
              'burns': burns.tolist(), **metrics}
    output['total_elapsed_s'] = initialization_s+metrics['elapsed_s']
    destination = ROOT/'results'/f'scvx_{args.case}_{args.initialization}_{args.objective}.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ('history', 'burns', 'times')}, indent=2))
    print(f'Saved {destination}')
    if not metrics['feasible']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
