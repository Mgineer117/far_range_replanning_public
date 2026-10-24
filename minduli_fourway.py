"""Shared execution and four-way disturbed-state recovery experiment."""

import json
import time

import numpy as np
import torch

import minduli_setting as m

METHODS = ('FRG', 'learned_only', 'checked_learned', 'checked_during_replanning')
FUEL_IMPROVEMENT_TOL = 1e-6  # m/s; ignore numerical differences at a tie.


def execute(start, t0, times, burns, stop=None):
    """Execute ECI impulses; a prefix excludes a burn exactly at handoff."""
    end = m.TOF if stop is None else stop
    x = m.to_eci(start, t0)
    trace, passive = [], []
    last = t0
    for t, burn in zip(times, burns):
        if stop is not None and t >= stop-1e-8:
            break
        if t > last+1e-8:
            trace.append((last, float(t), x.copy()))
            x = m.propagate(x, t-last)
        passive.append((float(t), x.copy()))
        x[3:] += burn
        last = float(t)
    if end > last+1e-8:
        trace.append((last, end, x.copy()))
        x = m.propagate(x, end-last)
    return x, trace, passive


def fixed_plan(fixed, start, t0, c, reference=None, start_after_burn=True):
    """Execute the same alpha/QCQP guidance against any selected reference."""
    _, _, trace = c.episode_fixed(fixed, start, t0, reference=reference,
                                 start_after_burn=start_after_burn)
    x = m.to_eci(start, t0)
    times, burns = [], []
    for t, nxt, after in trace:
        times.append(t)
        burns.append(after[3:]-x[3:])
        x = m.propagate(after, nxt-t)
    final_burn = c.NOM_BURNS[-1] if reference is None else reference[2][-1]
    return np.r_[times, m.TOF], np.vstack((burns, final_burn))


def learned_plan(planner, start, t0, c):
    times = np.linspace(t0, m.TOF, 6)
    nominal = m.trajectory(m.START, c.NOM_TIMES, c.NOM_BURNS, np.array([t0]))[0]
    raw, _, _ = c.policy_action(planner, c.observation(start, nominal, t0, (m.TOF-t0)/540),
                                 c.context(start, None, t0), False)
    first = np.clip(c.minimum_norm_seed(start, times)[:4]+10*np.tanh(raw).reshape(4, 3), -10, 10)
    return times, m.complete_last_two(start, times, first)[0][0]


def assess(start, t0, times, burns, sigma_limit, c):
    final, trace, passive = execute(start, t0, times, burns)
    pws = min(m.minimum_separation(x, t, nxt-t) for t, nxt, x in trace)
    pas = min(m.minimum_separation(x, t, 5400.) for t, x in passive)
    final = m.to_rel(final, m.TOF)
    sigma = c.covariance_sigma(trace)
    result = {
        'fuel_m_s': float(np.linalg.norm(burns, axis=1).sum()),
        'position_m': float(np.linalg.norm(final[:3]-m.GOAL[:3])),
        'velocity_m_s': float(np.linalg.norm(final[3:]-m.GOAL[3:])),
        'pws_m': pws, 'pas_m': pas,
        'max_burn_component_m_s': float(np.abs(burns).max()),
        'position_sigma_m': sigma,
        'covariance_limit_m': sigma_limit,
    }
    result['physical_feasible'] = bool(
        all(np.isfinite(v) for v in result.values()) and
        result['position_m'] < 1 and result['velocity_m_s'] < .01 and
        min(pws, pas) >= m.KOZ and result['max_burn_component_m_s'] <= 10+1e-7)
    result['feasible'] = bool(result['physical_feasible'] and sigma <= sigma_limit+1e-6)
    return result


def improvement_check(candidate, baseline):
    """Prefer feasibility, then strictly lower guided fuel on the same horizon."""
    saving = baseline['fuel_m_s']-candidate['fuel_m_s']
    fuel_improved = saving > FUEL_IMPROVEMENT_TOL
    if not candidate['feasible']:
        reason = 'candidate_infeasible'
    elif not baseline['feasible']:
        reason = 'feasible_recovery'
    elif fuel_improved:
        reason = 'fuel_improvement'
    else:
        reason = 'no_fuel_improvement'
    return {'accepted': reason in ('feasible_recovery', 'fuel_improvement'),
            'reason': reason, 'candidate_feasible': candidate['feasible'],
            'baseline_feasible': baseline['feasible'],
            'fuel_improved': fuel_improved, 'fuel_saving_m_s': saving}


def handoff_plan(start, t0, base, delay, iterations, sigma_limit, c, fixed):
    """Plan for a predicted handoff while the selected bridge is executed.

    shortcut: ideal dynamics make predicted and executed handoff states identical;
    add estimator/execution errors before drawing closed-loop robustness conclusions.
    """
    from minduli_scvx import scvx

    stop = t0+delay
    x, _, _ = execute(start, t0, *base, stop=stop)
    predicted = m.to_rel(x, stop)
    times = np.linspace(stop, m.TOF, 6)
    warm = c.minimum_norm_seed(predicted, times)
    times, burns, status = scvx(predicted, stop, warm=warm, iterations=iterations,
                                objective='fuel')
    reference = (predicted, times, burns)
    continuation = fixed_plan(fixed, predicted, stop, c, reference=reference)
    prefix = base[0] < stop-1e-8
    def with_prefix(plan):
        return (np.r_[base[0][prefix], plan[0]], np.vstack((base[1][prefix], plan[1])))

    combined = with_prefix(continuation)
    result = assess(start, t0, *combined, sigma_limit, c)
    # All counterfactuals share the executed bridge and its navigation covariance.
    nominal_continuation = fixed_plan(fixed, predicted, stop, c, start_after_burn=False)
    nominal_metrics = assess(start, t0, *with_prefix(nominal_continuation), sigma_limit, c)
    incumbent_metrics = assess(start, t0, *base, sigma_limit, c)
    nominal_check = improvement_check(result, nominal_metrics)
    incumbent_check = improvement_check(result, incumbent_metrics)
    spent_fuel = float(np.linalg.norm(base[1][prefix], axis=1).sum())
    # A computed plan is available only if it fits the explicitly modeled budget.
    deadline_met = status['elapsed_s'] <= delay
    continuation_metrics = assess(predicted, stop, *continuation, sigma_limit, c)
    switch = bool(status['feasible'] and continuation_metrics['physical_feasible'] and
                  deadline_met and nominal_check['accepted'] and incumbent_check['accepted'])
    if not status['feasible'] or not continuation_metrics['physical_feasible']:
        reason = 'infeasible_replanner'
    elif not deadline_met:
        reason = 'deadline_missed'
    elif not nominal_check['accepted']:
        reason = 'nominal_'+nominal_check['reason']
    elif not incumbent_check['accepted']:
        reason = 'incumbent_'+incumbent_check['reason']
    else:
        reason = incumbent_check['reason']
    info = {'replanner': 'SCvx-style', 'replanner_status': status['status'],
            'replanner_converged': status['converged'],
            'replanner_continuation_feasible': status['feasible'],
            'guided_continuation_feasible': continuation_metrics['physical_feasible'],
            'replanner_elapsed_s': status['elapsed_s'],
            'replanner_objective': 'fuel',
            'replanner_acceptance_reason': reason,
            'replanner_vs_nominal': nominal_check,
            'replanner_vs_incumbent': incumbent_check,
            'fuel_spent_before_handoff_m_s': spent_fuel,
            'candidate_remaining_fuel_m_s': result['fuel_m_s']-spent_fuel,
            'nominal_remaining_fuel_m_s': nominal_metrics['fuel_m_s']-spent_fuel,
            'incumbent_remaining_fuel_m_s': incumbent_metrics['fuel_m_s']-spent_fuel,
            'modeled_arrival_time_s': stop,
            'deadline_met_on_test_machine': deadline_met,
            'switched_to_replanner': switch}
    return (combined if switch else base), info


def summary(cases):
    common = [case for case in cases if all(case[name]['feasible'] for name in METHODS)]
    output = {}
    for name in METHODS:
        rows = [case[name] for case in cases]
        output[name] = {
            'cases': len(rows), 'feasible': sum(r['feasible'] for r in rows),
            'physical_feasible': sum(r['physical_feasible'] for r in rows),
            'pws_violations': sum(r['pws_m'] < m.KOZ for r in rows),
            'pas_violations': sum(r['pas_m'] < m.KOZ for r in rows),
            'burn_limit_violations': sum(r['max_burn_component_m_s'] > 10+1e-7 for r in rows),
            'covariance_rejections': sum(r['position_sigma_m'] > r['covariance_limit_m']+1e-6 for r in rows),
            'mean_fuel_all_m_s': float(np.mean([r['fuel_m_s'] for r in rows])),
            'mean_fuel_common_feasible_m_s': (float(np.mean([r[name]['fuel_m_s'] for r in common]))
                                               if common else None),
            'mean_position_sigma_m': float(np.mean([r['position_sigma_m'] for r in rows])),
        }
    paired = {}
    for name in METHODS[1:]:
        matched = [case for case in cases if case['FRG']['feasible'] and case[name]['feasible']]
        paired[name] = {'jointly_feasible_cases': len(matched),
                        'mean_fuel_saving_vs_FRG_m_s': (float(np.mean([
                            r['FRG']['fuel_m_s']-r[name]['fuel_m_s'] for r in matched]))
                            if matched else None)}
    return {'methods': output, 'common_feasible_cases': len(common),
            'pairwise_feasible_comparison': paired}


def run(data, test, out, delay=60., iterations=30):
    import minduli_compare as c

    c.NOM_TIMES, c.NOM_BURNS = data['nominal_times'], data['nominal_burns']
    if not np.isfinite(delay) or delay < 0 or any(data['replan_times'][test, 0]+delay >= m.TOF):
        raise ValueError('Planning delay must be nonnegative and leave time before arrival for every case')
    fixed, planner = c.FixedPolicy(), c.ReplanPolicy()
    fixed.load_state_dict(torch.load(out/'fixed_ppo.pt', map_location='cpu', weights_only=True))
    planner.load_state_dict(torch.load(out/'latent_ppo.pt', map_location='cpu', weights_only=True))
    fixed.eval(); planner.eval()
    config = {
        'protocol': 'shared-guidance-fuel-improvement-v3',
        'held_out_case_ids': [int(i) for i in test],
        'training': 'existing saved policies; no retraining',
        'state': 'true disturbed state; ideal thrust; nonlinear two-body propagation',
        'fallback_guidance': 'FRG',
        'planning_delay_s': delay,
        'delay_source': 'assumed simulation budget, NOT a measured flight-processor runtime',
        'replanner': 'SCvx-style, fixed six-node continuation, CW minimum-norm initialization',
        'replanner_iterations': iterations,
        'replanner_objective': 'fuel; physical feasibility and covariance independently screened',
        'acceptance': 'feasibility first; strictly lower guided fuel when the baseline is feasible',
        'fuel_improvement_tolerance_m_s': FUEL_IMPROVEMENT_TOL,
        'learned_comparator': 'original-reference guidance from the same disturbed state and time',
        'replanner_comparators': 'original-reference guidance and incumbent continuation from the same handoff state/time; same executed prefix and covariance history',
        'guidance': 'same saved FixedPolicy alpha selector and QCQP for every reference',
        'guidance_clock': 'original 540 s clock plus selected-reference feedforward burn nodes and handoff',
        'guidance_history': 'reset at disturbance and reference handoff; policy weights unchanged',
        'maneuver_schedule': 'learned/SCvx references each have six maneuver nodes; all are tracked by shared guidance',
        'handoff': 'plan from predicted bridge-end state; exact prediction under ideal dynamics',
        'deadline': 'no switch if measured SCvx wall time exceeds modeled delay',
        'candidate_check_overhead': 'measured but not inserted into simulated burn times',
        'criteria': {'position_m': 1., 'velocity_m_s': .01, 'separation_m': m.KOZ,
                     'eci_burn_component_m_s': 10.,
                     'covariance': 'final sigma no worse than nominal from the same disturbance phase'},
        'safety_validation': '10 s bracketing plus range-rate root refinement; no formal certificate',
        'covariance_measurements': 'common 10 s cadence from disturbance, plus final time',
        'learned_only': 'unscreened control baseline; nominal comparison recorded but not enforced',
        'checked_learned': 'accept feasible guided learned reference only if it improves on feasible FRG; feasible recovery allowed when FRG is infeasible',
        'checked_during_replanning': 'same learned selection, then switch only if feasible SCvx beats feasible nominal and incumbent continuations; otherwise retain incumbent',
    }
    output = {'setup': config, 'cases': {}}
    destination = out/'comparison_fourway.json'
    bounds = {}
    started = time.perf_counter()
    for number, case_id in enumerate(test, 1):
        i = int(case_id)
        start, t0 = data['starts'][i], float(data['replan_times'][i, 0])
        if t0 not in bounds:
            bounds[t0] = c.covariance_sigma(c.nominal_trace_from(t0))
        limit = bounds[t0]
        frg = fixed_plan(fixed, start, t0, c)
        frg_metrics = assess(start, t0, *frg, limit, c)
        tic = time.perf_counter()
        learned_reference = learned_plan(planner, start, t0, c)
        generation_s = time.perf_counter()-tic
        tic = time.perf_counter()
        learned = fixed_plan(fixed, start, t0, c, reference=(start, *learned_reference))
        learned_metrics = assess(start, t0, *learned, limit, c)
        checking_s = time.perf_counter()-tic
        learned_check = improvement_check(learned_metrics, frg_metrics)
        learned_metrics['comparison_to_nominal'] = learned_check
        accepted = learned_check['accepted']
        base = learned if accepted else frg
        checked = dict(learned_metrics if accepted else frg_metrics)
        checked['learned_accepted'] = accepted
        checked['learned_comparison_to_nominal'] = learned_check
        replacement, info = handoff_plan(start, t0, base, delay, iterations, limit, c, fixed)
        interim = assess(start, t0, *replacement, limit, c) if info['switched_to_replanner'] else dict(checked)
        interim.update(info, learned_accepted=accepted,
                       learned_comparison_to_nominal=learned_check,
                       bridge_guidance='learned' if accepted else 'FRG')
        rows = {'FRG': frg_metrics, 'learned_only': learned_metrics,
                'checked_learned': checked, 'checked_during_replanning': interim}
        output['cases'][str(i)] = rows
        rows['learned_only'].update(generation_elapsed_s=generation_s, checker_elapsed_s=checking_s)
        output.update(summary(list(output['cases'].values())))
        output['completed_cases'] = number
        output['elapsed_s'] = time.perf_counter()-started
        destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        print(f'Four-way {number}/{len(test)}: case {i}, candidate={accepted}, '
              f"switch={info['switched_to_replanner']}, solve={info['replanner_elapsed_s']:.2f}s", flush=True)
    output['learned_accepted'] = sum(v['checked_learned']['learned_accepted'] for v in output['cases'].values())
    output['replanner_switches'] = sum(v['checked_during_replanning']['switched_to_replanner'] for v in output['cases'].values())
    destination.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('cases','setup')}, indent=2))
    return output


def self_check(data):
    import minduli_compare as c

    c.NOM_TIMES, c.NOM_BURNS = data['nominal_times'], data['nominal_burns']
    feasible = {'feasible': True, 'fuel_m_s': 20.}
    cheaper = {'feasible': True, 'fuel_m_s': 19.}
    expensive = {'feasible': True, 'fuel_m_s': 30.}
    unsafe = {'feasible': False, 'fuel_m_s': 10.}
    assert improvement_check(cheaper, feasible)['reason'] == 'fuel_improvement'
    assert not improvement_check(expensive, feasible)['accepted']
    assert not improvement_check(feasible, feasible)['accepted']
    assert not improvement_check({'feasible': True, 'fuel_m_s': 20.-FUEL_IMPROVEMENT_TOL/2}, feasible)['accepted']
    assert not improvement_check(unsafe, feasible)['accepted']
    assert improvement_check(expensive, unsafe)['reason'] == 'feasible_recovery'
    i = 0
    start, times, burns = data['starts'][i], data['replan_times'][i], data['replan_burns'][i]
    final, trace, _ = execute(start, times[0], times, burns)
    expected = m.trajectory(start, times, burns, np.array([m.TOF]))[0]
    assert np.allclose(m.to_rel(final, m.TOF), expected, atol=1e-6)
    stop = times[2]
    x, _, _ = execute(start, times[0], times, burns, stop=stop)
    x[3:] += burns[2]
    expected = m.trajectory(start, times, burns, np.array([stop]))[0]
    assert np.allclose(m.to_rel(x, stop), expected, atol=1e-6)
    # Splitting a coast must not create additional navigation measurements.
    t, nxt, x = trace[0]
    split = t+1.234  # No measurement lies in this short first coast.
    split_trace = [(t, split, x), (split, nxt, m.propagate(x, split-t))]+trace[1:]
    assert np.isclose(c.covariance_sigma(trace), c.covariance_sigma(split_trace), rtol=1e-9)
    fixed = c.FixedPolicy()
    ft, fb = fixed_plan(fixed, start, times[0], c)
    _, original, _ = c.episode_fixed(fixed, start, times[0])
    final, _, _ = execute(start, times[0], ft, fb)
    result = m.to_rel(final, m.TOF)
    assert np.isclose(np.linalg.norm(result[:3]-m.GOAL[:3]), original['position_m'], atol=1e-5)
    assert np.isclose(np.linalg.norm(fb, axis=1).sum(), original['fuel_m_s'], atol=1e-8)
    guided = fixed_plan(fixed, start, times[0], c, reference=(start, times, burns))
    _, guided_trace, _ = execute(start, times[0], *guided)
    assert len(guided_trace) > 5  # Shared feedback updates, not six-burn replay.
    guided_final, _, _ = execute(start, times[0], *guided)
    assert np.linalg.norm(m.to_rel(guided_final, m.TOF)-m.GOAL) < 1e-3
    # A reference burn near a guidance tick must occur once, at its exact time.
    next_tick = (np.floor(times[0]/540.)+1)*540.
    nearby_times = times.copy(); nearby_times[1] = next_tick+.1
    nearby_burns = m.complete_last_two(start, nearby_times, burns[:4])[0][0]
    nearby = fixed_plan(fixed, start, times[0], c,
                        reference=(start, nearby_times, nearby_burns))
    tick = np.flatnonzero(np.isclose(nearby[0], next_tick, rtol=0, atol=1e-8))[0]
    burn = np.flatnonzero(np.isclose(nearby[0], next_tick+.1, rtol=0, atol=1e-8))[0]
    assert np.linalg.norm(nearby[1][tick]) < 1e-5
    assert np.allclose(nearby[1][burn], nearby_burns[1], atol=1e-5)
    # Starting away from the selected reference must produce a feedback correction.
    offset = start.copy(); offset[0] += 100.
    corrected = fixed_plan(fixed, offset, times[0], c, reference=(start, times, burns))
    assert not np.allclose(corrected[1][0], guided[1][0], atol=1e-7)
    base = (times, burns)
    retained, info = handoff_plan(start, times[0], base, 0., 1, 1e6, c, fixed)
    assert not info['deadline_met_on_test_machine'] and not info['switched_to_replanner']
    assert np.array_equal(retained[0], times) and np.array_equal(retained[1], burns)
    assert info['replanner_objective'] == 'fuel'
    prefix_fuel = info['fuel_spent_before_handoff_m_s']
    assert np.isclose(info['incumbent_remaining_fuel_m_s']+prefix_fuel,
                      np.linalg.norm(burns, axis=1).sum())
    # A handoff exactly at a nominal burn is before that burn, so apply it once.
    stop = c.NOM_TIMES[2]
    before = m.trajectory(m.START, c.NOM_TIMES, c.NOM_BURNS, np.array([stop]))[0]
    before[3:] -= m.target(stop)[1].T@c.NOM_BURNS[2]
    nominal = fixed_plan(fixed, before, stop, c, start_after_burn=False)
    assert np.allclose(nominal[1][0], c.NOM_BURNS[2], atol=1e-5)
    print('Shared guidance, improvement gate, handoff cost, covariance, and deadline checks passed')
