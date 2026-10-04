"""SAC (Soft Actor-Critic), adapté de CleanRL : `cleanrl/sac_continuous_action.py`.
Copyright (c) 2019 CleanRL developers, licence MIT. CleanRL l'a lui-même
adapté de plusieurs implémentations (haarnoja/sac, openai/spinningup,
pranz24/pytorch-soft-actor-critic, DLR-RM/stable-baselines3,
denisyarats/pytorch_sac) : leurs licences sont dans THIRD_PARTY_LICENSES.md.

Différences avec CleanRL : une mémoire de rejeu en numpy, un environnement
géré ici plutôt que `gym.vector`, la normalisation facultative des
observations (activée par défaut : les positions en mètres du football
dépassent largement 1), et une classe `Policy` : l'agent entraîné joue ensuite
en match avec `act`.

    from frondori import Agent
    from frondori.agents import SAC

    agent = SAC()
    Agent(environment="football-v0", local=True).train(agent, total_timesteps=1_000_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from ._common import Progress, TorchPolicy, torch

nn = torch.nn
F = torch.nn.functional

# Réglages par défaut de cleanrl/sac_continuous_action.py.
DEFAULTS = {
    "total_timesteps": 1_000_000,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "learning_starts": 5_000,
    "policy_lr": 3e-4,
    "q_lr": 1e-3,
    "policy_frequency": 2,
    "target_network_frequency": 1,
    "alpha": 0.2,
    "autotune": True,
    "hidden_sizes": (256, 256),
    "normalize_observations": True,
}

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class _SoftQNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        sizes = [observation_dim + action_dim, *hidden, 1]
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class _Actor(nn.Module):
    def __init__(self, observation_dim: int, action_space: spaces.Box, hidden: tuple[int, ...]):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        sizes = [observation_dim, *hidden]
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))
        self.fc_mean = nn.Linear(sizes[-1], action_dim)
        self.fc_logstd = nn.Linear(sizes[-1], action_dim)
        # Les actions de l'environnement vont de low à high : la sortie tanh
        # (-1..1) est remise à cette échelle.
        high, low = np.asarray(action_space.high, dtype=np.float32), np.asarray(action_space.low, dtype=np.float32)
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()  # astuce de reparamétrisation
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Correction due au tanh qui borne l'action.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC(TorchPolicy):
    """SAC pour actions `Box` bornées. En match, `act` joue l'action moyenne
    de la politique (déterministe) ; à l'entraînement, il échantillonne."""

    DEFAULTS = {key: None for key in DEFAULTS}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.actor = self.qf1 = self.qf2 = self.qf1_target = self.qf2_target = None
        self.log_alpha = None
        self.alpha = None

    def _defaults_for(self, action_space):
        return dict(DEFAULTS)

    def _check_action_space(self, action_space):
        if not isinstance(action_space, spaces.Box):
            raise ValueError(f"SAC gère les actions Box, pas {action_space} (essayer DQN ou PPO)")
        if not (np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high))):
            raise ValueError("SAC a besoin d'actions bornées (low et high finis)")

    def _build(self):
        hidden = tuple(self.config["hidden_sizes"])
        action_dim = int(np.prod(self.agent_action_space.shape))
        self.actor = _Actor(self.observation_dim, self.agent_action_space, hidden).to(self.device)
        self.qf1 = _SoftQNetwork(self.observation_dim, action_dim, hidden).to(self.device)
        self.qf2 = _SoftQNetwork(self.observation_dim, action_dim, hidden).to(self.device)
        self.qf1_target = _SoftQNetwork(self.observation_dim, action_dim, hidden).to(self.device)
        self.qf2_target = _SoftQNetwork(self.observation_dim, action_dim, hidden).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.config["q_lr"])
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.config["policy_lr"])
        if self.config["autotune"]:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config["q_lr"])
        else:
            self.alpha = self.config["alpha"]

    def _modules(self):
        return {
            "actor": self.actor,
            "qf1": self.qf1,
            "qf2": self.qf2,
            "qf1_target": self.qf1_target,
            "qf2_target": self.qf2_target,
        }

    def act(self, observation):
        self._require_trained()
        with torch.no_grad():
            action, _, mean = self.actor.get_action(self._tensor(self._prepare(observation)[None]))
        return (action if self.training else mean)[0].cpu().numpy()

    def learn(self, make_env, total_timesteps: int | None = None) -> list[dict]:
        """Entraîne l'agent ; renvoie le retour de chaque épisode."""
        env = self._make_envs(make_env, 1)[0]
        self._setup(env)
        c = self.config
        total_timesteps = int(total_timesteps or c["total_timesteps"])
        action_dim = int(np.prod(self.agent_action_space.shape))
        size = int(c["buffer_size"])
        buffer_obs = np.zeros((size, self.observation_dim), dtype=np.float32)
        buffer_next = np.zeros((size, self.observation_dim), dtype=np.float32)
        buffer_actions = np.zeros((size, action_dim), dtype=np.float32)
        buffer_rewards = np.zeros(size, dtype=np.float32)
        buffer_dones = np.zeros(size, dtype=np.float32)
        position, full = 0, False
        progress = Progress(total_timesteps, self.verbose)

        raw_obs, _ = env.reset()
        episode_return = 0.0
        for global_step in range(total_timesteps):
            if global_step < c["learning_starts"]:
                action = self.agent_action_space.sample()
            else:
                action = self.act(raw_obs)

            next_raw, reward, terminated, truncated, info = env.step(action)
            episode_return += info.get("env_reward", reward)
            buffer_obs[position], buffer_next[position] = raw_obs, next_raw
            buffer_actions[position], buffer_rewards[position] = np.ravel(action), reward
            buffer_dones[position] = float(terminated)
            position = (position + 1) % size
            full = full or position == 0
            if self.obs_rms is not None:
                self.obs_rms.update(next_raw)

            if terminated or truncated:
                progress.episode(global_step, episode_return)
                episode_return = 0.0
                raw_obs, _ = env.reset()
            else:
                raw_obs = next_raw

            if global_step > c["learning_starts"]:
                i = np.random.randint(0, size if full else position, size=int(c["batch_size"]))
                observations = self._tensor(self._prepare(buffer_obs[i]))
                next_observations = self._tensor(self._prepare(buffer_next[i]))
                actions = self._tensor(buffer_actions[i])
                rewards, dones = self._tensor(buffer_rewards[i]), self._tensor(buffer_dones[i])

                with torch.no_grad():
                    next_actions, next_log_pi, _ = self.actor.get_action(next_observations)
                    min_next_target = torch.min(
                        self.qf1_target(next_observations, next_actions), self.qf2_target(next_observations, next_actions)
                    ) - self.alpha * next_log_pi
                    next_q_value = rewards + (1 - dones) * c["gamma"] * min_next_target.view(-1)

                qf_loss = F.mse_loss(self.qf1(observations, actions).view(-1), next_q_value) + F.mse_loss(
                    self.qf2(observations, actions).view(-1), next_q_value
                )
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % c["policy_frequency"] == 0:
                    # Mise à jour retardée (comme TD3), compensée par
                    # `policy_frequency` mises à jour d'un coup.
                    for _ in range(int(c["policy_frequency"])):
                        pi, log_pi, _ = self.actor.get_action(observations)
                        min_qf_pi = torch.min(self.qf1(observations, pi), self.qf2(observations, pi))
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if c["autotune"]:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(observations)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                if global_step % c["target_network_frequency"] == 0:
                    for net, target in ((self.qf1, self.qf1_target), (self.qf2, self.qf2_target)):
                        for param, target_param in zip(net.parameters(), target.parameters()):
                            target_param.data.copy_(c["tau"] * param.data + (1 - c["tau"]) * target_param.data)

            progress.tick(global_step + 1)

        progress.tick(total_timesteps, force=True)
        env.close()
        return progress.history

    def _extra_state(self):
        return {"alpha": float(self.alpha) if self.alpha is not None else None}

    def _load_extra_state(self, state):
        if state.get("alpha") is not None:
            self.alpha = state["alpha"]
