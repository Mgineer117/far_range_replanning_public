"""Far-range rendezvous experiment in Wijayatunga et al.'s mission setting.

The nominal ECI burns are from Table 3 of the cited reference paper. The final two are
corrected for the table's four-decimal rounding. Recovery PSO uses the same
six-impulse, two-body, fixed-terminal-state formulation. This is an independent
implementation, not the authors' unavailable trained PPO checkpoint.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# Earth's gravitational parameter, G * Earth mass [m^3/s^2].
MU = 3.986004418e14
# Target's circular orbital radius: Earth radius + 300 km altitude [m].
R = 6_378_136.0 + 300_000.0
# Target's mean motion (orbital angular rate); orbital period = 2*pi/N [rad/s].
N = np.sqrt(MU / R**3)
# Nominal transfer time of flight: six hours [s].
TOF = 21_600.0
# Radius of the spherical keep-out zone centered on the target [m].
KOZ = 500.0
# Chaser states relative to the target in its rotating RTN frame:
# [radial, along-track, cross-track position (m), corresponding velocities (m/s)].
# Arrival is 1 km along-track, with zero velocity in the rotating frame.
GOAL = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
# Nominal initial state, approximately 100 km along-track from the target.
START = np.array([1.0, 100_000.0, 5.0, -0.02, 0.01, 0.0])
# Six nominal ECI-frame [x, y, z] velocity impulses from paper Table 3 [m/s].
# nominal() recomputes the final two burns to correct table rounding.
TABLE_BURNS = np.array([
    [-0.3222, -0.0691, 0.3964], [0.0, 0.0, 0.0],
    [-0.7197, 0.0313, -0.1560], [0.2889, -0.0495, 0.2627],
    [-0.0488, 0.5096, -2.9538], [-0.2966, 0.0456, -0.2554],
])
# Target orbit inclination, converted from degrees to radians.
INC = np.deg2rad(99.8)
# Fixed ECI unit vector along the ascending node; also target radial at t=0.
P = np.array([1.0, 0.0, 0.0])
# Fixed in-plane ECI unit vector 90 degrees ahead of P; target along-track at t=0.
Q = np.array([0.0, np.cos(INC), np.sin(INC)])
# Fixed unit orbit normal (angular momentum direction), completing the basis.
H = np.cross(P, Q)
# 1,000 sample times over the nominal transfer, including both endpoints [s].
GRID = np.linspace(0, TOF, 1000)
# Directory containing this script, used to locate data and result files.
HERE = Path(__file__).resolve().parent


def target(t):
    a = N * np.asarray(t)
    er = np.cos(a)[..., None] * P + np.sin(a)[..., None] * Q
    et = -np.sin(a)[..., None] * P + np.cos(a)[..., None] * Q
    en = np.broadcast_to(H, er.shape)
    c = np.stack((er, et, en), axis=-1)
    return np.concatenate((R * er, R * N * et), axis=-1), c


def to_eci(relative, t):
    relative = np.asarray(relative)
    xt, c = target(t)
    omega_cross_r = np.stack((-N * relative[..., 1], N * relative[..., 0],
                               np.zeros_like(relative[..., 0])), axis=-1)
    return xt + np.concatenate((np.einsum('...ij,...j->...i', c, relative[..., :3]),
                                np.einsum('...ij,...j->...i', c, relative[..., 3:] + omega_cross_r)), axis=-1)


def to_rel(eci, t):
    xt, c = target(t)
    d = np.asarray(eci) - xt
    r = np.einsum('...ji,...j->...i', c, d[..., :3])
    v = np.einsum('...ji,...j->...i', c, d[..., 3:])
    v -= np.stack((-N * r[..., 1], N * r[..., 0], np.zeros_like(r[..., 0])), axis=-1)
    return np.concatenate((r, v), axis=-1)


def propagate(x, dt):
    """Vectorized universal-variable, two-body Kepler propagation (SI units)."""
    x = np.asarray(x, dtype=float)
    dt = np.asarray(dt, dtype=float)
    r0, v0 = x[..., :3], x[..., 3:]
    radius = np.linalg.norm(r0, axis=-1)
    rv = np.sum(r0 * v0, axis=-1)
    alpha = 2 / radius - np.sum(v0 * v0, axis=-1) / MU
    chi = np.sqrt(MU) * alpha * dt
    def stumpff(z):
        z = np.asarray(z)
        positive = z > 1e-7
        negative = z < -1e-7
        s = np.sqrt(np.abs(z))
        c = np.where(positive, (1 - np.cos(s)) / np.where(positive, z, 1),
                     np.where(negative, (np.cosh(s) - 1) / np.where(negative, -z, 1),
                              .5 - z / 24 + z*z / 720))
        f = np.where(positive, (s - np.sin(s)) / np.where(positive, s**3, 1),
                     np.where(negative, (np.sinh(s) - s) / np.where(negative, s**3, 1),
                              1/6 - z / 120 + z*z / 5040))
        return c, f
    for _ in range(12):
        z = alpha * chi*chi
        c, s = stumpff(z)
        residual = rv/np.sqrt(MU)*chi*chi*c + (1-alpha*radius)*chi**3*s + radius*chi - np.sqrt(MU)*dt
        derivative = rv/np.sqrt(MU)*chi*(1-z*s) + (1-alpha*radius)*chi*chi*c + radius
        move = residual / derivative
        chi -= move
        if np.max(np.abs(move)) < 1e-8:
            break
    z = alpha * chi*chi
    c, s = stumpff(z)
    f = 1 - chi*chi/radius*c
    g = dt - chi**3/np.sqrt(MU)*s
    r = f[..., None]*r0 + g[..., None]*v0
    rnorm = np.linalg.norm(r, axis=-1)
    fdot = np.sqrt(MU)/(rnorm*radius)*(alpha*chi**3*s - chi)
    gdot = 1 - chi*chi/rnorm*c
    return np.concatenate((r, fdot[..., None]*r0 + gdot[..., None]*v0), axis=-1)


def complete_last_two(start, times, first_four, initial=None):
    """Terminal shooting: solve the penultimate burn, then match final velocity."""
    first_four = np.asarray(first_four)
    if first_four.ndim == 2:
        first_four = first_four[None]
    x = np.broadcast_to(to_eci(start, times[0]), (len(first_four), 6)).copy()
    for j in range(4):
        x[:, 3:] += first_four[:, j]
        x = propagate(x, times[j+1] - times[j])
    endpoint = to_eci(GOAL, times[-1])
    dt = times[-1] - times[-2]
    fifth = np.broadcast_to(np.zeros(3) if initial is None else initial, (len(x), 3)).copy()
    for _ in range(7):
        after = x.copy()
        after[:, 3:] += fifth
        end = propagate(after, dt)
        miss = endpoint[:3] - end[:, :3]
        if np.max(np.linalg.norm(miss, axis=1)) < 1e-4:
            break
        eps = .001
        probes = np.repeat(after[:, None], 3, axis=1)
        probes[:, np.arange(3), 3 + np.arange(3)] += eps
        shifted = propagate(probes, dt)
        jac = ((shifted[..., :3] - end[:, None, :3]) / eps).transpose(0, 2, 1)
        try:
            step = np.linalg.solve(jac, miss[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, miss[..., None], rcond=None)[0][..., 0]
        fifth += np.clip(step, -10, 10)
    after = x.copy()
    after[:, 3:] += fifth
    end = propagate(after, dt)
    sixth = endpoint[3:] - end[:, 3:]
    burns = np.concatenate((first_four, fifth[:, None], sixth[:, None]), axis=1)
    return burns, np.linalg.norm(end[:, :3]-endpoint[:3], axis=1)


def trajectory(start, times, burns, samples):
    """Propagate one six-burn plan at requested absolute times."""
    x = to_eci(start, times[0])
    out = np.empty((len(samples), 6))
    for j in range(5):
        x = x.copy()
        x[3:] += burns[j]
        mask = (samples >= times[j] - 1e-8) & (samples < times[j+1] - 1e-8)
        if np.any(mask):
            out[mask] = to_rel(propagate(x, samples[mask]-times[j]), samples[mask])
        x = propagate(x, times[j+1]-times[j])
    x[3:] += burns[-1]
    out[samples >= times[-1] - 1e-8] = to_rel(x, times[-1])
    return out


def nominal():
    times = np.linspace(0, TOF, 6)
    burns, miss = complete_last_two(START, times, TABLE_BURNS[:4], TABLE_BURNS[4])
    assert miss[0] < 1e-3
    path = trajectory(START, times, burns[0], GRID)
    assert np.linalg.norm(path[-1] - GOAL) < 1e-3
    return times, burns[0], path


def score(start, times, first_four, samples_per_segment=12):
    burns, terminal_miss = complete_last_two(start, times, first_four)
    count = len(burns)
    x = np.broadcast_to(to_eci(start, times[0]), (count, 6)).copy()
    fuel = np.linalg.norm(burns, axis=-1).sum(axis=1)
    obs = np.zeros(count)
    safety = np.zeros(count)
    min_pws = np.full(count, np.inf)
    min_pas = np.full(count, np.inf)
    for j in range(5):
        ballistic = x.copy()
        x[:, 3:] += burns[:, j]
        tt = times[j] + (times[j+1]-times[j]) * np.linspace(0, 1, samples_per_segment, endpoint=False)
        forced = propagate(x[:, None], tt[None]-times[j])
        free = propagate(ballistic[:, None], tt[None]-times[j])
        rel = to_rel(forced, tt[None])
        rel_free = to_rel(free, tt[None])
        rr = np.linalg.norm(rel[..., :3], axis=-1)
        min_pws = np.minimum(min_pws, rr.min(axis=1))
        safety += np.exp(-rr*rr/(2*(KOZ/3)**2)).sum(axis=1)
        y = rel[..., :3] / np.maximum(rr[..., None], 1e-9)
        free_rr = np.linalg.norm(rel_free[..., :3], axis=-1)
        y_free = rel_free[..., :3] / np.maximum(free_rr[..., None], 1e-9)
        obs += np.sum(y*y_free, axis=-1).mean(axis=1)
        # Direct ballistic-distance check. The paper uses an ROE-based RN
        # approximation; direct RN sampling labels even its nominal unsafe.
        pt = times[j] + np.linspace(0, 5400, 28)
        passive = to_rel(propagate(ballistic[:, None], pt[None]-times[j]), pt[None])
        pas = np.linalg.norm(passive[..., :3], axis=-1).min(axis=1)
        min_pas = np.minimum(min_pas, pas)
        safety += np.exp(-pas*pas/(2*(KOZ/3)**2))
        x = propagate(x, times[j+1]-times[j])
    # At the terminal node, paper uses the full distance for PAS.
    passive = x.copy()
    pt = times[-1] + np.linspace(0, 5400, 28)
    final_pas = to_rel(propagate(passive[:, None], pt[None]-times[-1]), pt[None])
    pas = np.linalg.norm(final_pas[..., :3], axis=-1).min(axis=1)
    min_pas = np.minimum(min_pas, pas)
    safety += np.exp(-pas*pas/(2*(KOZ/3)**2))
    overflow = np.maximum(np.abs(burns[:, 4:]) - 10, 0).sum(axis=(1, 2))
    # Obs mean has the same five segments for all candidates; weights from Table 2.
    value = fuel/1.7446 + obs*1000/.5 + safety + 100*overflow + terminal_miss
    # Keep the paper's soft terms, but require feasible recovery experts.
    # The original soft objective alone produced unsafe perturbed-state plans.
    violation = np.maximum(KOZ-min_pws, 0)/KOZ + np.maximum(KOZ-min_pas, 0)/KOZ
    value += 100_000*violation**2 + 100_000*overflow**2
    return value, burns, min_pws, min_pas


def pso(start, t0, seed, particles=16, iterations=20, points=12, warm=None):
    times = np.linspace(t0, TOF, 6)
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.5, 1.5, (particles, 4, 3))
    positions[0] = 0 if warm is None else warm
    velocities = np.zeros_like(positions)
    values, _, _, _ = score(start, times, positions, points)
    personal, personal_values = positions.copy(), values.copy()
    k = np.argmin(values)
    best, best_value = positions[k].copy(), values[k]
    for _ in range(iterations):
        velocities = .65*velocities + 1.4*rng.random(positions.shape)*(personal-positions)
        velocities += 1.4*rng.random(positions.shape)*(best-positions)
        positions = np.clip(positions+velocities, -10, 10)
        values, _, _, _ = score(start, times, positions, points)
        better = values < personal_values
        personal[better], personal_values[better] = positions[better], values[better]
        k = np.argmin(personal_values)
        if personal_values[k] < best_value:
            best, best_value = personal[k].copy(), personal_values[k]
    value, burns, pws, pas = score(start, times, best[None], max(points, 50))
    return times, burns[0], float(value[0]), float(pws[0]), float(pas[0])


def minimum_separation(eci, t, duration):
    """Refine closest approaches on one unforced coast; not a formal certificate."""
    from scipy.optimize import brentq

    if not np.isfinite(duration) or duration < 0:
        raise ValueError('Coast duration must be finite and nonnegative')
    grid = np.linspace(0, duration, max(2, int(np.ceil(duration/10))+1))
    states = to_rel(propagate(eci, grid), t+grid)
    rate = np.einsum('ij,ij->i', states[:, :3], states[:, 3:])
    distances = np.linalg.norm(states[:, :3], axis=1)
    if not np.all(np.isfinite(states)):
        raise ValueError('Nonfinite coast state')
    minimum = float(distances.min())

    def radial_rate(dt):
        state = to_rel(propagate(eci, dt), t+dt)
        return float(np.dot(state[:3], state[3:]))

    for j in np.flatnonzero((rate[:-1] < 0) & (rate[1:] > 0)):
        dt = brentq(radial_rate, grid[j], grid[j+1], xtol=1e-5)
        state = to_rel(propagate(eci, dt), t+dt)
        minimum = min(minimum, float(np.linalg.norm(state[:3])))
    return minimum


def validated_safety(start, times, burns):
    """Refine PWS and missed-burn passive closest approaches on each coast."""
    x = to_eci(start, times[0])
    pws, pas = np.inf, np.inf
    for j, t in enumerate(times):
        pas = min(pas, minimum_separation(x, t, 5400))
        x = x.copy()
        x[3:] += burns[j]
        if j < 5:
            dt = times[j+1]-t
            pws = min(pws, minimum_separation(x, t, dt))
            x = propagate(x, dt)
    return pws, pas


def generate(path, points=100, per_point=10, particles=16, iterations=20, seed=7,
             position_perturbation=15_000., velocity_perturbation=0.):
    nominal_times, nominal_burns, nominal_path = nominal()
    rng = np.random.default_rng(seed)
    ids = np.linspace(30, 900, points, dtype=int)
    starts, indices, controls, burn_times, plans, metrics = [], [], [], [], [], []
    checkpoint = path.with_suffix('.partial.npz')
    first_group = 0
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if (int(saved['points']) != points or int(saved['per_point']) != per_point or
                float(saved['position_perturbation']) != position_perturbation or
                float(saved['velocity_perturbation']) != velocity_perturbation):
                raise ValueError('Checkpoint settings differ from requested generation settings')
            starts = list(saved['starts'])
            indices = list(saved['indices'])
            controls = list(saved['replan_burns'])
            burn_times = list(saved['replan_times'])
            metrics = list(saved['replan_metrics'])
            offsets = saved['replan_offsets']
            all_paths = saved['replan_paths']
            plans = [all_paths[offsets[j]:offsets[j+1]] for j in range(len(starts))]
            first_group = len(starts)//per_point
            rng.bit_generator.state = json.loads(str(saved['rng_state']))
        print(f'resuming {len(starts)} plans from {checkpoint}', flush=True)

    def save(destination):
        offsets = np.r_[0, np.cumsum([len(p) for p in plans])]
        np.savez_compressed(destination, grid=GRID, nominal=nominal_path, nominal_times=nominal_times,
                            nominal_burns=nominal_burns, starts=starts, indices=indices,
                            replan_times=burn_times, replan_burns=controls,
                            replan_offsets=offsets, replan_paths=np.concatenate(plans),
                            replan_metrics=metrics, seed=seed, points=points, per_point=per_point,
                            position_perturbation=position_perturbation,
                            velocity_perturbation=velocity_perturbation,
                            rng_state=json.dumps(rng.bit_generator.state))
    tic = time.time()
    for g, k in enumerate(ids[first_group:], start=first_group):
        for q in range(per_point):
            scale = min(1., max(.1, np.linalg.norm(nominal_path[k, :3])/np.linalg.norm(START[:3])))
            for attempt in range(30):
                if attempt % 3 == 0:
                    position_spread = position_perturbation*np.array([1., 1., .05])*scale
                    delta = np.r_[rng.uniform(-position_spread, position_spread),
                                  rng.uniform(-velocity_perturbation, velocity_perturbation, 3)]
                    state = nominal_path[k] + delta
                    warm = None
                size = particles if attempt % 3 == 0 else max(particles, 32)
                rounds = iterations if attempt % 3 == 0 else max(iterations, 50)
                times, burns, cost, _, _ = pso(state, GRID[k], seed+g*per_point+q+10_000*attempt,
                                                size, rounds, warm=warm)
                pws, pas = validated_safety(state, times, burns)
                terminal = trajectory(state, times, burns, np.array([TOF]))[0]
                if (min(pws, pas) >= KOZ and np.max(np.abs(burns)) <= 10 and
                    np.linalg.norm(terminal-GOAL) < 1e-3):
                    break
                warm = burns[:4]
            else:
                raise RuntimeError(f'No feasible plan at nominal index {k}; increase PSO budget')
            starts.append(state)
            indices.append(k)
            controls.append(burns)
            burn_times.append(times)
            plans.append(trajectory(state, times, burns, GRID[k:]))
            metrics.append((cost, pws, pas))
        print(f'{g+1}/{points} nominal points; {len(starts)} plans; {time.time()-tic:.1f}s', flush=True)
        if (g+1)%10 == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            save(checkpoint)
    save(path)
    checkpoint.unlink(missing_ok=True)
    print(f'saved {path}', flush=True)


def self_check():
    x = to_eci(START, 0)
    assert np.allclose(to_rel(x, 0), START, atol=1e-8)
    assert np.allclose(propagate(propagate(x, 1234), 4321), propagate(x, 5555), atol=1e-5)
    times, burns, path = nominal()
    assert np.linalg.norm(path[-1]-GOAL) < 1e-3
    assert abs(np.linalg.norm(burns, axis=1).sum()-5.0381) < .01
    # Regression: all coarse samples pass although the coast enters the KOZ.
    mid = 4.25
    near_target = np.array([497.4, 0., 0., 0., 30., 0.])
    coast = propagate(to_eci(near_target, mid), -mid)
    coarse_times = np.linspace(0, 17., 3)
    coarse = to_rel(propagate(coast, coarse_times), coarse_times)
    assert np.linalg.norm(coarse[:, :3], axis=1).min() > KOZ
    assert abs(minimum_separation(coast, 0., 17.)-497.4) < 1e-3
    print(json.dumps({'nominal_fuel_m_s': float(np.linalg.norm(burns, axis=1).sum()),
                      'nominal_terminal_error_m': float(np.linalg.norm(path[-1, :3]-GOAL[:3])),
                      'nominal_min_pws_m': float(np.linalg.norm(path[:, :3], axis=1).min())}, indent=2))


def audit(path):
    with np.load(path) as saved:
        count = len(saved['starts'])
        final = saved['replan_paths'][saved['replan_offsets'][1:]-1]
        terminal_error = np.linalg.norm(final-GOAL, axis=1)
        safety = saved['replan_metrics'][:, 1:3]
        burn_max = np.max(np.abs(saved['replan_burns']), axis=(1, 2))
        assert count == int(saved['points'])*int(saved['per_point'])
        assert np.max(terminal_error) < 1e-3
        assert np.min(safety) >= KOZ
        assert np.max(burn_max) <= 10
        print(json.dumps({'plans':count, 'max_terminal_state_error':float(terminal_error.max()),
                          'minimum_pws_m':float(safety[:,0].min()),
                          'minimum_pas_m':float(safety[:,1].min()),
                          'maximum_burn_component_m_s':float(burn_max.max())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['self-check', 'generate', 'audit'])
    parser.add_argument('--output', type=Path, default=HERE/'data/minduli_setting_data.npz')
    parser.add_argument('--points', type=int, default=100)
    parser.add_argument('--per-point', type=int, default=10)
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--position-perturbation', type=float, default=15_000.)
    parser.add_argument('--velocity-perturbation', type=float, default=0.)
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check()
    elif args.command == 'audit':
        audit(args.output)
    else:
        generate(args.output, args.points, args.per_point, args.particles, args.iterations,
                 position_perturbation=args.position_perturbation,
                 velocity_perturbation=args.velocity_perturbation)


if __name__ == '__main__':
    main()
