"""PPO (Proximal Policy Optimization), adapté de CleanRL : `cleanrl/ppo.py`
(actions discrètes) et `cleanrl/ppo_continuous_action.py` (actions continues).
Copyright (c) 2019 CleanRL developers, licence MIT (cf. THIRD_PARTY_LICENSES.md).

Différences avec CleanRL : un seul agent pour les deux types d'action (les
réglages par défaut de chaque script CleanRL selon l'action), des
environnements parallèles gérés ici (une simple liste, réinitialisée à la fin
de chaque épisode) plutôt que `gym.vector`, et une classe `Policy` : l'agent
entraîné joue ensuite en match avec `act`.

    from frondori import Agent
    from frondori.agents import PPO

    agent = PPO()
    Agent(environment="kitchen-v0", local=True).train(agent, total_timesteps=500_000)
    agent.save("ppo.pt")
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from ._common import Progress, RunningMeanStd, TorchPolicy, mlp, torch

nn = torch.nn
Categorical = torch.distributions.Categorical
Normal = torch.distributions.Normal

COMMON = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "anneal_lr": True,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "hidden_sizes": (64, 64),
}
# Réglages par défaut de cleanrl/ppo.py.
DISCRETE = {
    **COMMON,
    "total_timesteps": 500_000,
    "learning_rate": 2.5e-4,
    "num_envs": 4,
    "num_steps": 128,
    "num_minibatches": 4,
    "update_epochs": 4,
    "ent_coef": 0.01,
    "normalize_observations": False,
    "normalize_rewards": False,
}
# Réglages par défaut de cleanrl/ppo_continuous_action.py (dont ses
# wrappers : normalisation des observations et des récompenses).
CONTINUOUS = {
    **COMMON,
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "num_envs": 1,
    "num_steps": 2048,
    "num_minibatches": 32,
    "update_epochs": 10,
    "ent_coef": 0.0,
    "normalize_observations": True,
    "normalize_rewards": True,
}


class _Network(nn.Module):
    def __init__(self, observation_dim: int, action_space: spaces.Space, hidden: tuple[int, ...]):
        super().__init__()
        self.discrete = isinstance(action_space, spaces.Discrete)
        self.critic = mlp([observation_dim, *hidden, 1], nn.Tanh, last_std=1.0)
        if self.discrete:
            self.actor = mlp([observation_dim, *hidden, int(action_space.n)], nn.Tanh, last_std=0.01)
        else:
            action_dim = int(np.prod(action_space.shape))
            self.actor = mlp([observation_dim, *hidden, action_dim], nn.Tanh, last_std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def value(self, x):
        return self.critic(x)

    def distribution(self, x):
        if self.discrete:
            return Categorical(logits=self.actor(x))
        mean = self.actor(x)
        return Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))

    def action_and_value(self, x, action=None):
        distribution = self.distribution(x)
        if action is None:
            action = distribution.sample()
        log_prob, entropy = distribution.log_prob(action), distribution.entropy()
        if not self.discrete:
            log_prob, entropy = log_prob.sum(1), entropy.sum(1)
        return action, log_prob, entropy, self.critic(x)


class PPO(TorchPolicy):
    """PPO pour actions `Discrete` ou `Box`.

    Chaque réglage laissé à `None` prend la valeur par défaut de CleanRL pour
    le type d'action de l'environnement (cf. `DISCRETE`, `CONTINUOUS`).

    `deterministic` : en match, jouer l'action la plus probable (la moyenne
    en continu) plutôt que de la tirer selon la politique.
    `wrap_env` : appliqué à chaque environnement d'entraînement (un wrapper
    Gymnasium, par exemple pour façonner la récompense).
    """

    DEFAULTS = {key: None for key in {**DISCRETE, **CONTINUOUS}}

    def __init__(self, *, deterministic: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.deterministic = deterministic
        self.network: _Network | None = None
        self.optimizer = None

    def _defaults_for(self, action_space):
        return dict(DISCRETE if isinstance(action_space, spaces.Discrete) else CONTINUOUS)

    def _check_action_space(self, action_space):
        if not isinstance(action_space, (spaces.Discrete, spaces.Box)):
            raise ValueError(f"PPO handles Discrete and Box actions, not {action_space}")

    def _build(self):
        self.network = _Network(self.observation_dim, self.agent_action_space, tuple(self.config["hidden_sizes"])).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"], eps=1e-5)

    def _modules(self):
        return {"network": self.network}

    def act(self, observation):
        self._require_trained()
        x = self._tensor(self._prepare(observation)[None])
        with torch.no_grad():
            distribution = self.network.distribution(x)
            if self.deterministic and not self.training:
                action = distribution.probs.argmax(1) if self.network.discrete else distribution.mean
            else:
                action = distribution.sample()
        if self.network.discrete:
            return int(action.item())
        return np.clip(action[0].cpu().numpy(), self.agent_action_space.low, self.agent_action_space.high)

    def learn(self, make_env, total_timesteps: int | None = None) -> list[dict]:
        """Entraîne l'agent ; renvoie le retour de chaque épisode
        (`[{"step": ..., "return": ...}, ...]`)."""
        first = self._make_envs(make_env, 1)[0]
        self._setup(first)
        c = self.config
        total_timesteps = int(total_timesteps or c["total_timesteps"])
        num_envs, num_steps = int(c["num_envs"]), int(c["num_steps"])
        batch_size = num_envs * num_steps
        minibatch_size = batch_size // int(c["num_minibatches"])
        num_iterations = max(1, total_timesteps // batch_size)
        discrete = self.network.discrete
        action_space = self.agent_action_space

        envs = [first] + self._make_envs(make_env, num_envs - 1)
        progress = Progress(num_iterations * batch_size, self.verbose)

        action_shape = () if discrete else action_space.shape
        obs = torch.zeros((num_steps, num_envs, self.observation_dim), device=self.device)
        actions = torch.zeros((num_steps, num_envs) + action_shape, device=self.device)
        logprobs = torch.zeros((num_steps, num_envs), device=self.device)
        rewards = torch.zeros((num_steps, num_envs), device=self.device)
        dones = torch.zeros((num_steps, num_envs), device=self.device)
        values = torch.zeros((num_steps, num_envs), device=self.device)

        # Normalisation des récompenses (comme gymnasium.wrappers.NormalizeReward) :
        # divisées par l'écart-type glissant du retour actualisé.
        return_rms = RunningMeanStd(()) if c["normalize_rewards"] else None
        discounted = np.zeros(num_envs)

        raw_obs = np.stack([env.reset()[0] for env in envs])
        episode_returns = np.zeros(num_envs)
        next_done = torch.zeros(num_envs, device=self.device)
        global_step = 0

        for iteration in range(1, num_iterations + 1):
            if c["anneal_lr"]:
                self.optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / num_iterations) * c["learning_rate"]

            for step in range(num_steps):
                global_step += num_envs
                next_obs = self._tensor(self._prepare(raw_obs, update=True))
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = self.network.action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                step_actions = action.cpu().numpy()
                if not discrete:  # comme gymnasium.wrappers.ClipAction
                    step_actions = np.clip(step_actions, action_space.low, action_space.high)
                step_rewards, step_dones, next_raw = np.zeros(num_envs), np.zeros(num_envs), []
                for i, env in enumerate(envs):
                    o, r, terminated, truncated, info = env.step(step_actions[i])
                    episode_returns[i] += info.get("env_reward", r)
                    step_rewards[i] = r
                    if terminated or truncated:
                        progress.episode(global_step, float(episode_returns[i]))
                        episode_returns[i] = 0.0
                        step_dones[i] = 1.0
                        o, _ = env.reset()
                    next_raw.append(o)
                raw_obs = np.stack(next_raw)

                if return_rms is not None:
                    discounted = discounted * c["gamma"] + step_rewards
                    return_rms.update(discounted)
                    step_rewards = np.clip(step_rewards / np.sqrt(return_rms.var + 1e-8), -10, 10)
                    discounted[step_dones == 1.0] = 0.0
                rewards[step] = self._tensor(step_rewards)
                next_done = self._tensor(step_dones)

            # Avantages par GAE, en bootstrappant sur la dernière observation.
            with torch.no_grad():
                next_value = self.network.value(self._tensor(self._prepare(raw_obs))).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal, nextvalues = 1.0 - next_done, next_value
                    else:
                        nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                    delta = rewards[t] + c["gamma"] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + c["gamma"] * c["gae_lambda"] * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, self.observation_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            indices = np.arange(batch_size)
            for _epoch in range(int(c["update_epochs"])):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start : start + minibatch_size]
                    _, newlogprob, entropy, newvalue = self.network.action_and_value(
                        b_obs[mb], b_actions[mb].long() if discrete else b_actions[mb]
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb]
                    if c["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        -mb_advantages * torch.clamp(ratio, 1 - c["clip_coef"], 1 + c["clip_coef"]),
                    ).mean()

                    newvalue = newvalue.view(-1)
                    if c["clip_vloss"]:
                        v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -c["clip_coef"], c["clip_coef"])
                        v_loss = 0.5 * torch.max((newvalue - b_returns[mb]) ** 2, (v_clipped - b_returns[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                    loss = pg_loss - c["ent_coef"] * entropy.mean() + v_loss * c["vf_coef"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), c["max_grad_norm"])
                    self.optimizer.step()

                if c["target_kl"] is not None and approx_kl > c["target_kl"]:
                    break

            progress.tick(global_step)

        progress.tick(global_step, force=True)
        for env in envs:
            env.close()
        return progress.history
