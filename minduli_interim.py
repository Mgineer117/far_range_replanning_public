"""Provisional burn during PSO replanning on the held-out rendezvous cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import minduli_compare as c
import minduli_setting as m

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'minduli_setting_data.npz'
STEP = 540.


def predicted_plan(surrogate, data, i):
    start, times = data['starts'][i], data['replan_times'][i]
    t0 = float(times[0])
    nominal = m.trajectory(m.START, c.NOM_TIMES, c.NOM_BURNS, np.array([t0]))[0]
    obs = c.observation(start, nominal, t0, (m.TOF-t0)/STEP)
    raw, _, _ = c.policy_action(surrogate, obs, c.context(start, None, t0), False)
    first_four = np.clip(c.minimum_norm_seed(start, times)[:4] +
                         10*np.tanh(raw).reshape(4, 3), -10, 10)
    burns, _ = m.complete_last_two(start, times, first_four)
    return times, burns[0]


def assess_plan(start, times, burns):
    pws, pas = m.validated_safety(start, times, burns)
    final = m.trajectory(start, times, burns, np.array([m.TOF]))[0]
    return {'pso_fuel_m_s': float(np.linalg.norm(burns, axis=1).sum()),
            'pso_pws_m': pws, 'pso_pas_m': pas,
            'terminal_position_m': float(np.linalg.norm(final[:3]-m.GOAL[:3])),
            'terminal_velocity_m_s': float(np.linalg.norm(final[3:]-m.GOAL[3:])),
            'pso_max_burn_component_m_s': float(np.abs(burns).max())}


def plan_valid(metrics):
    return (metrics['terminal_position_m'] < 1 and
            metrics['terminal_velocity_m_s'] < .01 and
            min(metrics['pso_pws_m'], metrics['pso_pas_m']) >= m.KOZ and
            metrics['pso_max_burn_component_m_s'] <= 10)


def continuation(start, t0, seed):
    """Finite-budget PSO from a handoff state, with feasible retries."""
    best = None
    for particles, iterations, offset in ((16, 20, 0), (32, 50, 100_000),
                                           (64, 100, 200_000),
                                           (64, 100, 10_000),
                                           (64, 100, 20_000),
                                           (64, 100, 30_000)):
        times, burns, score, _, _ = m.pso(start, t0, seed+offset,
                                          particles=particles, iterations=iterations)
        result = (times, burns, {**assess_plan(start, times, burns),
                                 'pso_score': score})
        if best is None or score < best[2]['pso_score']:
            best = result
        if plan_valid(result[2]):
            return result, True
    return best, False


def same_pipeline_case(policy, start, t0, proposed, seed, delay=.1078):
    """One immediate interim burn, then PSO from its reached state."""
    if proposed is False:
        burn = np.zeros(3)
    elif proposed is None:
        reference = m.trajectory(m.START, c.NOM_TIMES, c.NOM_BURNS,
                                 np.array([t0]))[0]
        next_guidance = (np.floor(t0/STEP)+1)*STEP
        obs = c.observation(start, reference, t0,
                            int(np.ceil((m.TOF-t0)/STEP)))
        raw, _, _ = c.policy_action(policy, obs, np.zeros(48, np.float32), False)
        correction, _ = c.qcqp(start-reference, next_guidance-t0,
                                float(np.clip(raw[0]+1, 0, 2)))
        correction *= min(1, 10/(np.max(np.abs(correction))+1e-12))
        burn = m.target(t0)[1]@correction
    else:
        burn = proposed[1][0].copy()
    x = m.to_eci(start, t0)
    x[3:] += burn
    handoff_t = t0+delay
    handoff = m.to_rel(m.propagate(x, delay), handoff_t)
    interim_times = t0+np.linspace(0, delay, 3)
    interim_path = m.to_rel(m.propagate(x, interim_times-t0), interim_times)
    plan, valid = continuation(handoff, handoff_t, seed)
    times, burns, result = plan
    trace = [(t0, handoff_t, x.copy())]
    x = m.to_eci(handoff, handoff_t)
    for j in range(5):
        x = x.copy()
        x[3:] += burns[j]
        trace.append((times[j], times[j+1], x.copy()))
        x = m.propagate(x, times[j+1]-times[j])
    return {**result,
            'total_fuel_m_s': float(np.linalg.norm(burn)+result['pso_fuel_m_s']),
            'minimum_pws_m': min(float(np.linalg.norm(interim_path[:, :3], axis=1).min()),
                                 result['pso_pws_m']),
            'minimum_pas_m': min(c.passivity(m.to_eci(start, t0)+np.r_[np.zeros(3), burn],
                                            t0, samples=120), result['pso_pas_m']),
            'interim_burn_m_s': float(np.linalg.norm(burn)),
            'interim_max_component_m_s': float(np.max(np.abs(burn))),
            'pso_valid': valid}, trace


def same_pipeline():
    """Evaluate an active interim burn on the main 200-case, no-error split."""
    with np.load(DATA) as file:
        data = {key: file[key] for key in file.files}
    c.NOM_TIMES, c.NOM_BURNS = data['nominal_times'], data['nominal_burns']
    policy = c.FixedPolicy()
    policy.load_state_dict(torch.load(RESULTS/'fixed_ppo.pt', weights_only=True))
    surrogate = c.ReplanPolicy()
    surrogate.load_state_dict(torch.load(RESULTS/'latent_ppo.pt', weights_only=True))
    surrogate.eval()
    test = np.flatnonzero((np.arange(len(data['starts']))//int(data['per_point']))%5 == 0)
    assert len(test) == 200
    output = {'setting': 'same 200 held-out disturbances, no thrust error',
              'planning_delay_s': .1078,
              'delay_source': 'fixed modeled delay from a prior median PSO solve',
              'cases': {}}
    path = RESULTS/'interim_same_pipeline.json'
    for n, i in enumerate(test, 1):
        i = int(i)
        start, t0 = data['starts'][i], float(data['replan_times'][i, 0])
        proposed = predicted_plan(surrogate, data, i)
        waiting, waiting_trace = same_pipeline_case(policy, start, t0, False, 17+i)
        fixed, fixed_trace = same_pipeline_case(policy, start, t0, None, 17+i)
        interim, interim_trace = same_pipeline_case(policy, start, t0, proposed, 17+i)
        if n % 10 == 1:
            waiting['final_position_sigma_m'] = c.covariance_sigma(waiting_trace)
            fixed['final_position_sigma_m'] = c.covariance_sigma(fixed_trace)
            interim['final_position_sigma_m'] = c.covariance_sigma(interim_trace)
        output['cases'][str(i)] = {'wait_for_pso': waiting,
                                   'capped_fixed_reference': fixed,
                                   'learned_provisional_reference': interim}
        if n % 10 == 0:
            print(f'Completed {n}/200 same-pipeline handoffs', flush=True)
            path.write_text(json.dumps(output, indent=2,
                                       default=lambda value: value.item()))
    for name in ('wait_for_pso', 'capped_fixed_reference',
                 'learned_provisional_reference'):
        rows = [case[name] for case in output['cases'].values()]
        output[name] = {
            'mean_total_fuel_m_s': float(np.mean([r['total_fuel_m_s'] for r in rows])),
            'median_total_fuel_m_s': float(np.median([r['total_fuel_m_s'] for r in rows])),
            'mean_interim_burn_m_s': float(np.mean([r['interim_burn_m_s'] for r in rows])),
            'pws_violations': sum(r['minimum_pws_m'] < m.KOZ for r in rows),
            'pas_violations': sum(r['minimum_pas_m'] < m.KOZ for r in rows),
            'burn_limit_cases': sum(r['interim_max_component_m_s'] > 10+1e-8 or
                                    r['pso_max_burn_component_m_s'] > 10+1e-8 for r in rows),
            'feasible_cases': sum(r['pso_valid'] and r['terminal_position_m'] < 1 and
                                  r['terminal_velocity_m_s'] < .01 and
                                  r['minimum_pws_m'] >= m.KOZ and
                                  r['minimum_pas_m'] >= m.KOZ and
                                  r['interim_max_component_m_s'] <= 10+1e-8 for r in rows),
            'final_position_sigma_20_cases_m': float(np.mean([
                r['final_position_sigma_m'] for r in rows if 'final_position_sigma_m' in r]))}
    old = np.array([v['capped_fixed_reference']['total_fuel_m_s']
                    for v in output['cases'].values()])
    new = np.array([v['learned_provisional_reference']['total_fuel_m_s']
                    for v in output['cases'].values()])
    output['paired_fuel_saving_m_s'] = float(np.mean(old-new))
    output['learned_cheaper_cases'] = int(np.sum(new < old))
    waiting = np.array([v['wait_for_pso']['total_fuel_m_s']
                        for v in output['cases'].values()])
    output['learned_vs_wait_fuel_saving_m_s'] = float(np.mean(waiting-new))
    output['learned_cheaper_than_wait_cases'] = int(np.sum(new < waiting))
    phase_savings = (waiting-new).reshape(-1, int(data['per_point'])).mean(axis=1)
    samples = np.random.default_rng(31).choice(
        phase_savings, (10_000, len(phase_savings)), replace=True).mean(axis=1)
    output['learned_vs_wait_phase_bootstrap95_m_s'] = np.percentile(
        samples, [2.5, 97.5]).tolist()
    path.write_text(json.dumps(output, indent=2,
                               default=lambda value: value.item()))
    print(json.dumps({key: value for key, value in output.items() if key != 'cases'},
                     indent=2, default=lambda value: value.item()))


def self_check():
    with np.load(DATA) as file:
        data = {key: file[key] for key in file.files}
    c.NOM_TIMES, c.NOM_BURNS = data['nominal_times'], data['nominal_burns']
    policy = c.FixedPolicy()
    surrogate = c.ReplanPolicy()
    surrogate.load_state_dict(torch.load(RESULTS/'latent_ppo.pt', weights_only=True))
    i = 0
    start, t0 = data['starts'][i], float(data['replan_times'][i, 0])
    proposed = predicted_plan(surrogate, data, i)
    waiting, _ = same_pipeline_case(policy, start, t0, False, 17+i)
    interim, _ = same_pipeline_case(policy, start, t0, proposed, 17+i)
    assert waiting['interim_burn_m_s'] == 0
    assert np.isfinite(interim['total_fuel_m_s'])
    print('Provisional guidance check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check() if args.self_check else same_pipeline()
