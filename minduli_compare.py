"""Compare reconstructed fixed-reference PPO/QCQP with learned replanning.

Run after ``minduli_setting.py generate``. Both controllers use the same
nonlinear mission dynamics and the same held-out disturbance states. The
fixed-reference QCQP uses CW, the circular-orbit limit of the YA STM.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

import minduli_setting as m


torch.set_num_threads(1)
SCALE = np.array([100_000.0]*3 + [5.0]*3)
DEV_SCALE = np.array([15_000.0]*3 + [1.0]*3)


def cw(dt):
    """Clohessy--Wiltshire state transition for relative position and velocity."""
    dt = np.asarray(dt, dtype=float)
    n = m.N
    c, s = np.cos(n*dt), np.sin(n*dt)
    a = np.zeros(dt.shape + (6, 6))
    a[..., 0, 0], a[..., 0, 3], a[..., 0, 4] = 4-3*c, s/n, 2*(1-c)/n
    a[..., 1, 0], a[..., 1, 1] = 6*(s-n*dt), 1
    a[..., 1, 3], a[..., 1, 4] = 2*(c-1)/n, (4*s-3*n*dt)/n
    a[..., 2, 2], a[..., 2, 5] = c, s/n
    a[..., 3, 0], a[..., 3, 3], a[..., 3, 4] = 3*n*s, c, 2*s
    a[..., 4, 0], a[..., 4, 3], a[..., 4, 4] = 6*n*(c-1), -2*s, 4*c-3
    a[..., 5, 2], a[..., 5, 5] = -n*s, c
    return a


def observation(state, nominal, t, remaining, prev_reward=0, prev_alpha=1):
    absolute = m.to_eci(state, t)
    return np.clip(np.r_[remaining/40, absolute[:3]/m.R, absolute[3:]/(m.R*m.N),
                         state/SCALE, prev_reward/1000, prev_alpha,
                         (state-nominal)/DEV_SCALE], -10, 10).astype(np.float32)


def context(state, nominal_path, t):
    future = np.linspace(t, m.TOF, 8)
    states = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, future)
    return np.clip(((states-state)/DEV_SCALE).ravel(), -10, 10).astype(np.float32)


def qcqp(deviation, dt, alpha):
    a = cw(dt)
    error = a @ deviation
    b = a[:, 3:]
    radius = alpha*np.linalg.norm(deviation)
    if np.linalg.norm(error) <= radius + 1e-8:
        return np.zeros(3), True
    left, singular, right = np.linalg.svd(b, full_matrices=False)
    projection = left.T @ error
    floor = max(0, error@error - projection@projection)
    if radius*radius <= floor + 1e-12:
        return -right.T @ (projection/singular), floor <= radius*radius + 1e-8
    def residual(lam):
        return floor + np.sum((projection/(1+lam*singular**2))**2)
    lo, hi = 0., 1e-12
    while residual(hi) > radius*radius:
        hi *= 2
    for _ in range(42):
        mid = (lo+hi)/2
        if residual(mid) > radius*radius:
            lo = mid
        else:
            hi = mid
    return -right.T @ (hi*singular*projection/(1+hi*singular**2)), True


class FixedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        def net():
            layers, width = [], 21
            for _ in range(4):
                layers.extend((nn.Linear(width, 64), nn.Tanh()))
                width = 64
            return nn.Sequential(*layers, nn.Linear(width, 1))
        self.actor, self.critic = net(), net()
        self.log_std = nn.Parameter(torch.tensor([np.log(.25)], dtype=torch.float32))

    def forward(self, obs, ctx=None):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs, ctx=None):
        mean, value = self(obs)
        return Normal(mean, self.log_std.clamp(-3, .5).exp()), value


class ReplanPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = nn.Sequential(nn.Linear(21, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 8), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(40, 64), nn.Tanh())
        self.actor = nn.Linear(64, 12)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((12,), -3.0))

    def forward(self, obs, ctx):
        h = self.head(torch.cat((self.original(obs), self.context(ctx)), dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs, ctx):
        mean, value = self(obs, ctx)
        return Normal(mean, self.log_std.clamp(-3, .5).exp()), value


def passivity(eci, t, last=False, samples=20):
    times = t + np.linspace(0, 5400, samples)
    states = m.to_rel(m.propagate(eci, times-t), times)
    return float(np.linalg.norm(states[:, :3], axis=1).min())


def execute_error(eci_burn, t, rng, level):
    if not level:
        return eci_burn
    c = m.target(t)[1]
    u = c.T @ eci_burn
    mag = np.linalg.norm(u)
    if mag < 1e-12:
        return eci_burn
    sigma_mag, sigma_ang = ((.02, 1) if level == 'low' else (.05, 2))
    beta = np.arcsin(np.clip(u[2]/mag, -1, 1)) + np.deg2rad(rng.normal(0, sigma_ang))
    gamma = np.arctan2(u[0], u[1]) + np.deg2rad(rng.normal(0, sigma_ang))
    magnitude = mag*(1+rng.normal(0, sigma_mag))
    return c @ (magnitude*np.array([np.cos(beta)*np.sin(gamma),
                                     np.cos(beta)*np.cos(gamma), np.sin(beta)]))


def policy_action(policy, obs, ctx, stochastic):
    obs_t = torch.from_numpy(obs)[None]
    ctx_t = torch.from_numpy(ctx)[None]
    with torch.no_grad():
        distribution, value = policy.distribution(obs_t, ctx_t)
        raw = distribution.sample()[0] if stochastic else distribution.mean[0]
        logp = distribution.log_prob(raw[None]).sum().item()
    return raw.numpy(), logp, value.item()


def episode_fixed(policy, start, t0, stochastic=False, constant=None, error=None, seed=0,
                  start_after_burn=True, enforce_limit=False, ctx=None):
    rng = np.random.default_rng(seed)
    nodes = np.r_[t0, np.arange(np.floor(t0/540)+1, 41)*540.]
    nodes = np.unique(np.clip(nodes, t0, m.TOF))
    x = m.to_eci(start, t0)
    fuel = 0.
    pws, pas = np.inf, np.inf
    records = []
    trace = []
    prev_reward, prev_alpha = 0., 1.
    infeasible = 0
    impulse_limit_violations = 0
    for j, (t, nxt) in enumerate(zip(nodes[:-1], nodes[1:])):
        burn_id = np.flatnonzero(np.isclose(NOM_TIMES, t, atol=1e-5))
        nominal_burn = NOM_BURNS[burn_id[0]] if len(burn_id) else np.zeros(3)
        nominal_post = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([t]))[0]
        if len(burn_id) and t == t0 and start_after_burn:
            # Disturbance states are sampled after any burn at their nominal point.
            nominal_burn = np.zeros(3)
        x = x.copy()
        x[3:] += nominal_burn
        current = m.to_rel(x, t)
        obs = observation(current, nominal_post, t, len(nodes)-j-1, prev_reward, prev_alpha)
        step_ctx = np.zeros(48, np.float32) if ctx is None else ctx
        raw, logp, value = policy_action(policy, obs, step_ctx, stochastic)
        alpha = 0. if j == len(nodes)-2 else (constant if constant is not None else np.clip(raw[0]+1, 0, 2))
        correction, feasible = qcqp(current-nominal_post, nxt-t, alpha)
        infeasible += not feasible
        impulse_limit_violations += np.max(np.abs(correction)) > 10
        if enforce_limit:
            correction *= min(1, 10/(np.max(np.abs(correction))+1e-12))
        commanded = nominal_burn + m.target(t)[1] @ correction
        executed = execute_error(commanded, t, rng, error)
        # nominal_burn is already in x, so apply only the difference plus correction.
        x[3:] += executed - nominal_burn
        trace.append((t, nxt, x.copy()))
        fuel += np.linalg.norm(commanded)
        x_next = m.propagate(x, nxt-t)
        if not stochastic:
            tt = np.linspace(0, nxt-t, 20)
            rel = m.to_rel(m.propagate(x, tt), t+tt)
            pws = min(pws, np.linalg.norm(rel[:, :3], axis=1).min())
        local_pas = passivity(x_next, nxt, nxt == m.TOF, 20 if stochastic else 120)
        pas = min(pas, local_pas)
        next_rel = m.to_rel(x_next, nxt)
        range_next = np.linalg.norm(next_rel[:3])
        safety = 1000*max(0, m.KOZ-range_next, m.KOZ-local_pas)
        nominal_pre = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([nxt]))[0]
        next_burn = np.flatnonzero(np.isclose(NOM_TIMES, nxt, atol=1e-5))
        if len(next_burn):
            nominal_pre[3:] -= m.target(nxt)[1].T @ NOM_BURNS[next_burn[0]]
        # Eq. (12) observability at the next node, compared to nominal.
        free_rel = m.to_rel(m.propagate(x-np.r_[np.zeros(3), executed], nxt-t), nxt)
        nom_free = m.to_rel(m.propagate(m.to_eci(nominal_post, t)-np.r_[np.zeros(3), nominal_burn], nxt-t), nxt)
        def cosine(a, b):
            return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
        eta = cosine(free_rel[:3], next_rel[:3])
        eta_nom = cosine(nom_free[:3], nominal_pre[:3])
        obs_penalty = 1000*max(0, eta-eta_nom)
        reward = -(np.linalg.norm(commanded)+safety+obs_penalty)
        if stochastic:
            records.append((obs, step_ctx, raw.astype(np.float32), logp, value, reward/1000))
        prev_reward, prev_alpha = reward, alpha
        x = x_next
    # Final nominal burn at arrival, matching the paper's six-impulse schedule.
    final_command = NOM_BURNS[-1]
    x[3:] += execute_error(final_command, m.TOF, rng, error)
    fuel += np.linalg.norm(final_command)
    final = m.to_rel(x, m.TOF)
    return records, {'fuel_m_s': fuel, 'position_m': np.linalg.norm(final[:3]-m.GOAL[:3]),
                     'velocity_m_s': np.linalg.norm(final[3:]-m.GOAL[3:]),
                     'pws_m': pws, 'pas_m': pas, 'infeasible_steps': infeasible,
                     'correction_limit_violations': impulse_limit_violations}, trace


def expert_path(data, i, times):
    return m.trajectory(data['starts'][i], data['replan_times'][i], data['replan_burns'][i], times)


def minimum_norm_seed(start, times):
    # CW is the circular-orbit limit of the paper's YA STM.
    influence = np.concatenate([cw(m.TOF-t)[:, 3:]@m.target(t)[1].T for t in times], axis=1)
    target = m.GOAL-cw(m.TOF-times[0])@start
    return (influence.T@np.linalg.solve(influence@influence.T, target)).reshape(6, 3)


def episode_replan(policy, data, i, stochastic=False, error=None):
    start = data['starts'][i]
    times = data['replan_times'][i]
    nominal = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([times[0]]))[0]
    obs = observation(start, nominal, times[0], (m.TOF-times[0])/540)
    ctx = context(start, None, times[0])
    raw, logp, value = policy_action(policy, obs, ctx, stochastic)
    first_four = np.clip(minimum_norm_seed(start, times)[:4]+10*np.tanh(raw).reshape(4, 3), -10, 10)
    burns, miss = m.complete_last_two(start, times, first_four)
    burns = burns[0].copy()
    x = m.to_eci(start, times[0])
    pws, pas = np.inf, np.inf
    trace = []
    rng = np.random.default_rng(17+i)
    path_times = np.linspace(times[0], m.TOF, 50)
    path = []
    for j, t in enumerate(times):
        pas = min(pas, passivity(x, t, j == 5, 20 if stochastic else 120))
        if error and j == 4:
            # Recompute the terminal pair from the state reached after earlier thrust errors.
            pair_times = np.r_[np.full(5, t), m.TOF]
            pair, _ = m.complete_last_two(m.to_rel(x, t), pair_times, np.zeros((4, 3)))
            burns[4:] = pair[0, 4:]
        if error and j == 5:
            burns[5] = m.to_eci(m.GOAL, m.TOF)[3:]-x[3:]
        x = x.copy()
        x[3:] += execute_error(burns[j], t, rng, error)
        if j < 5:
            trace.append((t, times[j+1], x.copy()))
            mask = (path_times >= t-1e-8)&(path_times < times[j+1]-1e-8)
            if np.any(mask):
                segment = m.to_rel(m.propagate(x, path_times[mask]-t), path_times[mask])
                path.append(segment)
            safety_times = t + np.linspace(0, times[j+1]-t, 20 if stochastic else 120)
            safety_path = m.to_rel(m.propagate(x, safety_times-t), safety_times)
            pws = min(pws, np.linalg.norm(safety_path[:, :3], axis=1).min())
            x = m.propagate(x, times[j+1]-t)
    path.append(m.to_rel(x, m.TOF)[None])
    path = np.concatenate(path)
    target = expert_path(data, i, path_times)
    tracking = float(np.linalg.norm(path[:, :3]-target[:, :3], axis=1).mean())
    final = m.to_rel(x, m.TOF)
    fuel = float(np.linalg.norm(burns, axis=1).sum())
    component_overflow = float(np.maximum(np.abs(burns)-10, 0).sum())
    safety = max(0, m.KOZ-pws, m.KOZ-pas)/m.KOZ
    reward = -(tracking/1000 + .05*fuel + 100*safety + 100*component_overflow)
    records = [(obs, ctx, raw.astype(np.float32), logp, value, reward/10)] if stochastic else []
    return records, {'fuel_m_s': fuel, 'position_m': np.linalg.norm(final[:3]-m.GOAL[:3]),
                     'velocity_m_s': np.linalg.norm(final[3:]-m.GOAL[3:]),
                     'pws_m': pws, 'pas_m': pas, 'tracking_m': tracking,
                     'burn_limit_violation_m_s': component_overflow}, trace


def covariance_sigma(trace):
    """Angles-only EKF covariance check at 10 s; returns final 1-sigma position (m)."""
    p = np.diag(np.array([300., 100., 100., .3, .3, .3])**2)
    q = np.diag(np.array([50., 50., 50., .1, .1, .1])**2)
    sensor = np.eye(3)*1e-6
    ident = np.eye(6)
    last_t = trace[0][0]
    for t, nxt, x in trace:
        sample_times = np.arange(max(t+10, last_t+10), nxt+1e-6, 10.)
        if not len(sample_times) or sample_times[-1] < nxt-1e-6:
            sample_times = np.r_[sample_times, nxt]
        states = m.to_rel(m.propagate(x, sample_times-t), sample_times)
        for ts, state in zip(sample_times, states):
            dt = ts-last_t
            a = cw(dt)
            p = a@p@a.T + q*(dt/10)
            r = state[:3]
            distance = np.linalg.norm(r)
            y = r/distance
            h = np.zeros((3, 6))
            h[:, :3] = (np.eye(3)-np.outer(y, y))/distance
            hp = h@p
            gain = np.linalg.solve(hp@h.T+sensor, hp).T
            kh = ident-gain@h
            p = kh@p@kh.T + gain@sensor@gain.T
            last_t = ts
    return float(np.sqrt(np.linalg.eigvalsh(p[:3, :3]).max()))


def expert_trace(data, i):
    times = data['replan_times'][i]
    burns = data['replan_burns'][i]
    x = m.to_eci(data['starts'][i], times[0])
    trace = []
    for j in range(5):
        x = x.copy()
        x[3:] += burns[j]
        trace.append((times[j], times[j+1], x.copy()))
        x = m.propagate(x, times[j+1]-times[j])
    return trace


def nominal_trace_from(t0):
    nodes = np.r_[t0, NOM_TIMES[NOM_TIMES > t0+1e-6]]
    x = m.to_eci(m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([t0]))[0], t0)
    trace = []
    for t, nxt in zip(nodes[:-1], nodes[1:]):
        burn = np.flatnonzero(np.isclose(NOM_TIMES, t, atol=1e-6))
        if len(burn):
            x[3:] += NOM_BURNS[burn[0]]
        trace.append((t, nxt, x.copy()))
        x = m.propagate(x, nxt-t)
    return trace


def ppo_train(policy, rollout, train_ids, validation_ids, updates, episodes, seed, out,
              history_path=None):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.003 if isinstance(policy, FixedPolicy) else .0002)
    def validation():
        ms = [rollout(policy, int(i), False)[1] for i in validation_ids[:min(20, len(validation_ids))]]
        return float(np.mean([v['position_m']/1000+v['velocity_m_s']/.5+.05*v['fuel_m_s']+
                              100*(v['pws_m']<m.KOZ)+100*(v['pas_m']<m.KOZ)+
                              100*v.get('burn_limit_violation_m_s', 0) for v in ms]))
    best, best_update, best_weights = validation(), 0, copy.deepcopy(policy.state_dict())
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(f'update,validation_cost,best_cost\n0,{best:.6f},{best:.6f}\n')
    for update in range(updates):
        rows, advs, returns = [], [], []
        for i in rng.choice(train_ids, size=episodes, replace=True):
            episode = rollout(policy, int(i), True)[0]
            rewards = np.array([row[-1] for row in episode], np.float32)
            values = np.array([row[-2] for row in episode]+[0], np.float32)
            gae = 0.
            episode_adv = np.empty(len(episode), np.float32)
            for j in range(len(episode)-1, -1, -1):
                gae = rewards[j]+.99*values[j+1]-values[j]+.99*gae
                episode_adv[j] = gae
            rows.extend(episode)
            advs.extend(episode_adv)
            returns.extend(episode_adv+values[:-1])
        obs = torch.tensor(np.stack([r[0] for r in rows]))
        ctx = torch.tensor(np.stack([r[1] for r in rows]))
        actions = torch.tensor(np.stack([r[2] for r in rows]))
        old_logp = torch.tensor([r[3] for r in rows], dtype=torch.float32)
        adv = torch.tensor(np.asarray(advs))
        adv = (adv-adv.mean())/(adv.std()+1e-6)
        targets = torch.tensor(np.asarray(returns))
        for _ in range(10):
            for batch in torch.randperm(len(rows)).split(64):
                dist, value = policy.distribution(obs[batch], ctx[batch])
                ratio = (dist.log_prob(actions[batch]).sum(-1)-old_logp[batch]).exp()
                objective = torch.minimum(ratio*adv[batch], ratio.clamp(.9, 1.1)*adv[batch])
                loss = -objective.mean()+.5*(value-targets[batch]).square().mean()-.01*dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step()
        if (update+1) % 5 == 0 or update+1 == updates:
            score = validation()
            if score < best:
                best, best_update, best_weights = score, update+1, copy.deepcopy(policy.state_dict())
            if history_path is not None:
                with history_path.open('a') as stream:
                    stream.write(f'{update+1},{score:.6f},{best:.6f}\n')
            print(f'{out.stem}: PPO {update+1}/{updates}, validation={score:.2f}, best={best:.2f}', flush=True)
    policy.load_state_dict(best_weights)
    torch.save(best_weights, out)
    return best_update


def imitation(policy, data, train_ids, epochs=80):
    rows = []
    for i in train_ids:
        t = data['replan_times'][i, 0]
        state = data['starts'][i]
        nominal = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([t]))[0]
        burns = data['replan_burns'][i, :4]
        seed = minimum_norm_seed(state, data['replan_times'][i])[:4]
        rows.append((observation(state, nominal, t, (m.TOF-t)/540), context(state, None, t),
                     np.arctanh(np.clip(((burns-seed)/10).ravel(), -.999, .999)).astype(np.float32)))
    obs, ctx, action = [torch.tensor(np.stack([r[j] for r in rows])) for j in range(3)]
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    for epoch in range(epochs):
        for batch in torch.randperm(len(rows)).split(256):
            raw, _ = policy(obs[batch], ctx[batch])
            loss = (10*raw.tanh()-10*action[batch].tanh()).square().sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'cloning {epoch+1}/{epochs}, action MSE={loss.item():.3f}', flush=True)


def self_check():
    global NOM_TIMES, NOM_BURNS
    NOM_TIMES, NOM_BURNS, _ = m.nominal()
    delta = np.array([1500., -600., 20., 0., 0., 0.])
    control, feasible = qcqp(delta, 540., .7)
    assert feasible
    assert np.linalg.norm(cw(540.)@np.r_[delta[:3], delta[3:]+control]) <= .7*np.linalg.norm(delta)+1e-5
    start = m.START + np.array([5000., -1000., 200., 0., 0., 0.])
    times = np.linspace(4320., m.TOF, 6)
    burns, miss = m.complete_last_two(start, times, minimum_norm_seed(start, times)[:4])
    assert miss[0] < 1e-3 and np.isfinite(burns).all()
    print('QCQP contraction and nonlinear terminal solve checks passed')


def summarize(rows):
    result = {key: {'mean': float(np.mean([row[key] for row in rows])),
                    'median': float(np.median([row[key] for row in rows]))}
              for key in rows[0]}
    result['pws_violations'] = int(sum(row['pws_m'] < m.KOZ for row in rows))
    result['pas_violations'] = int(sum(row['pas_m'] < m.KOZ for row in rows))
    result['feasible_cases'] = int(sum(
        row['position_m'] < 1 and row['velocity_m_s'] < .01 and
        row['pws_m'] >= m.KOZ and row['pas_m'] >= m.KOZ and
        row.get('burn_limit_violation_m_s', 0) < 1e-6 and
        row.get('correction_limit_violations', 0) == 0 for row in rows))
    return result


def train(data, train_ids, validation_ids, out, args):
    """Train the reconstructed fixed-reference controller and learned planner."""
    torch.manual_seed(7)
    paper_fixed = FixedPolicy()
    with torch.no_grad():
        paper_fixed.actor[-1].weight.zero_()
        paper_fixed.actor[-1].bias.fill_(-.3543)
    rng = np.random.default_rng(73)
    spread = np.array([15_000., 15_000., 750., 0., 0., 0.])
    paper_states = np.vstack((m.START + rng.uniform(-spread, spread, (700, 6)),
                              [m.START + sign*.5*np.eye(6)[j]*spread
                               for sign in (1, -1) for j in range(6)]))
    paper_rollout = lambda policy, i, stochastic: episode_fixed(
        policy, paper_states[i], 0., stochastic, start_after_burn=False)
    ppo_train(paper_fixed, paper_rollout, np.arange(700), np.arange(700, 712),
              args.paper_updates, args.episodes, 73, out/'paper_fixed_ppo.pt')
    fixed = FixedPolicy()
    fixed.load_state_dict(paper_fixed.state_dict())
    fixed_rollout = lambda policy, i, stochastic: episode_fixed(
        policy, data['starts'][i], m.GRID[int(data['indices'][i])], stochastic)
    ppo_train(fixed, fixed_rollout, train_ids, validation_ids, args.updates,
              args.episodes, 7, out/'fixed_ppo.pt')
    planner = ReplanPolicy()
    imitation(planner, data, train_ids, args.epochs)
    planner_rollout = lambda policy, i, stochastic: episode_replan(
        policy, data, i, stochastic)
    ppo_train(planner, planner_rollout, train_ids, validation_ids, args.updates,
              args.episodes, 9, out/'latent_ppo.pt')


def evaluate(data, test, out):
    fixed = FixedPolicy()
    fixed.load_state_dict(torch.load(out/'fixed_ppo.pt', weights_only=True))
    planner = ReplanPolicy()
    planner.load_state_dict(torch.load(out/'latent_ppo.pt', weights_only=True))
    fixed.eval()
    planner.eval()
    output = {}
    fixed_rows = [episode_fixed(fixed, data['starts'][i],
                  m.GRID[int(data['indices'][i])])[1] for i in test]
    direct_rows = [episode_replan(planner, data, int(i))[1] for i in test]
    output['FRG'] = summarize(fixed_rows)
    output['DLR'] = summarize(direct_rows)
    selected, accepted, costs = [], 0, []
    nominal_covariance = {}
    for i in test:
        i = int(i)
        _, learned, trace = episode_replan(planner, data, i)
        start, times = data['starts'][i], data['replan_times'][i]
        t0 = float(times[0])
        nominal = m.trajectory(m.START, NOM_TIMES, NOM_BURNS, np.array([t0]))[0]
        obs = observation(start, nominal, t0, (m.TOF-t0)/540)
        raw, _, _ = policy_action(planner, obs, context(start, None, t0), False)
        first_four = np.clip(minimum_norm_seed(start, times)[:4] +
                             10*np.tanh(raw).reshape(4, 3), -10, 10)
        burns, _ = m.complete_last_two(start, times, first_four)
        pws, pas = m.validated_safety(start, times, burns[0])
        learned['pws_m'], learned['pas_m'] = pws, pas
        valid = (learned['position_m'] < 1 and learned['velocity_m_s'] < .01 and
                 pws >= m.KOZ and pas >= m.KOZ and
                 learned['burn_limit_violation_m_s'] < 1e-6)
        if valid:
            if t0 not in nominal_covariance:
                nominal_covariance[t0] = covariance_sigma(nominal_trace_from(t0))
            valid = covariance_sigma(trace) <= nominal_covariance[t0]
        if valid:
            selected.append(learned)
            accepted += 1
            costs.append((m.score(start, times, first_four[None], 50)[0][0],
                          m.score(start, times, data['replan_burns'][i,:4][None], 50)[0][0]))
        else:
            terminal = data['replan_paths'][data['replan_offsets'][i+1]-1]
            selected.append({'fuel_m_s': float(np.linalg.norm(data['replan_burns'][i], axis=1).sum()),
                             'position_m': float(np.linalg.norm(terminal[:3]-m.GOAL[:3])),
                             'velocity_m_s': float(np.linalg.norm(terminal[3:]-m.GOAL[3:])),
                             'pws_m': float(data['replan_metrics'][i, 1]),
                             'pas_m': float(data['replan_metrics'][i, 2])})
    keys = ('fuel_m_s', 'position_m', 'velocity_m_s', 'pws_m', 'pas_m')
    output['VLRR'] = summarize([{key: row[key] for key in keys} for row in selected])
    output['VLRR']['accepted_learned'] = accepted
    output['VLRR']['stored_pso_fallback'] = len(test)-accepted
    if costs:
        output['VLRR']['accepted_learned_objective_mean'] = float(np.mean(costs, axis=0)[0])
        output['VLRR']['same_case_pso_objective_mean'] = float(np.mean(costs, axis=0)[1])
    (out/'comparison.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--data', type=Path, default=m.HERE/'data/minduli_setting_data.npz')
    parser.add_argument('--output', type=Path, default=m.HERE/'results')
    parser.add_argument('--updates', type=int, default=40)
    parser.add_argument('--paper-updates', type=int, default=80)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as file:
        data = {key: file[key] for key in file.files}
    global NOM_TIMES, NOM_BURNS
    NOM_TIMES, NOM_BURNS = data['nominal_times'], data['nominal_burns']
    groups = np.arange(len(data['starts']))//int(data['per_point'])
    train_ids = np.flatnonzero((groups%5 != 0)&(groups%10 != 1))
    validation_ids = np.flatnonzero(groups%10 == 1)
    test = np.flatnonzero(groups%5 == 0)
    if args.train:
        train(data, train_ids, validation_ids, args.output, args)
    evaluate(data, test, args.output)


if __name__ == '__main__':
    main()
