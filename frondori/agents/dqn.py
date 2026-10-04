"""DQN (Deep Q-Network), adapté de CleanRL : `cleanrl/dqn.py`.
Copyright (c) 2019 CleanRL developers, licence MIT (cf. THIRD_PARTY_LICENSES.md).

Différences avec CleanRL : une mémoire de rejeu en numpy (au lieu de celle de
Stable-Baselines3), un environnement géré ici plutôt que `gym.vector`, et une
classe `Policy` : l'agent entraîné joue ensuite en match avec `act`.

    from frondori import Agent
    from frondori.agents import DQN

    agent = DQN()
    Agent(environment="kitchen-v0", local=True).train(agent, total_timesteps=500_000)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from gymnasium import spaces

from ._common import Progress, TorchPolicy, mlp, torch

nn = torch.nn
F = torch.nn.functional

# Réglages par défaut de cleanrl/dqn.py.
DEFAULTS = {
    "total_timesteps": 500_000,
    "learning_rate": 2.5e-4,
    "buffer_size": 10_000,
    "gamma": 0.99,
    "tau": 1.0,
    "target_network_frequency": 500,
    "batch_size": 128,
    "start_e": 1.0,
    "end_e": 0.05,
    "exploration_fraction": 0.5,
    "learning_starts": 10_000,
    "train_frequency": 10,
    "hidden_sizes": (120, 84),
    "normalize_observations": False,
}


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ReplayBuffer:
    """Mémoire circulaire des transitions."""

    def __init__(self, size: int, observation_dim: int):
        self.size = size
        self.observations = np.zeros((size, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((size, observation_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.position = 0
        self.full = False

    def add(self, observation, next_observation, action, reward, done) -> None:
        i = self.position
        self.observations[i], self.next_observations[i] = observation, next_observation
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.position = (i + 1) % self.size
        self.full = self.full or self.position == 0

    def sample(self, batch_size: int) -> np.ndarray:
        return np.random.randint(0, self.size if self.full else self.position, size=batch_size)


class DQN(TorchPolicy):
    """DQN pour actions `Discrete`.

    `play_epsilon` : part d'actions au hasard en match (0 par défaut). Un peu
    de hasard peut éviter que deux agents coopératifs entièrement
    déterministes se bloquent l'un l'autre.
    """

    DEFAULTS = {key: None for key in DEFAULTS}

    def __init__(self, *, play_epsilon: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.play_epsilon = play_epsilon
        # Exploration courante, utilisée par `act` pendant l'entraînement (y
        # compris par l'adversaire en self-play).
        self.epsilon = 1.0
        self.q_network = None
        self.target_network = None
        self.optimizer = None

    def _defaults_for(self, action_space):
        return dict(DEFAULTS)

    def _check_action_space(self, action_space):
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(f"DQN handles Discrete actions, not {action_space} (try SAC or PPO)")

    def _build(self):
        sizes = [self.observation_dim, *self.config["hidden_sizes"], int(self.agent_action_space.n)]
        self.q_network = mlp(sizes, nn.ReLU).to(self.device)
        self.target_network = mlp(sizes, nn.ReLU).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config["learning_rate"])

    def _modules(self):
        return {"q_network": self.q_network, "target_network": self.target_network}

    def act(self, observation):
        self._require_trained()
        epsilon = self.epsilon if self.training else self.play_epsilon
        if random.random() < epsilon:
            return int(self.agent_action_space.sample())
        with torch.no_grad():
            q_values = self.q_network(self._tensor(self._prepare(observation)[None]))
        return int(q_values.argmax(1).item())

    def learn(self, make_env, total_timesteps: int | None = None) -> list[dict]:
        """Entraîne l'agent ; renvoie le retour de chaque épisode."""
        env = self._make_envs(make_env, 1)[0]
        self._setup(env)
        c = self.config
        total_timesteps = int(total_timesteps or c["total_timesteps"])
        buffer = ReplayBuffer(int(c["buffer_size"]), self.observation_dim)
        progress = Progress(total_timesteps, self.verbose)

        raw_obs, _ = env.reset()
        episode_return = 0.0
        for global_step in range(total_timesteps):
            self.epsilon = linear_schedule(c["start_e"], c["end_e"], c["exploration_fraction"] * total_timesteps, global_step)
            action = self.act(raw_obs)

            next_raw, reward, terminated, truncated, info = env.step(action)
            episode_return += info.get("env_reward", reward)
            # Stockées brutes : la normalisation (si active) s'applique au
            # tirage, avec les statistiques du moment.
            buffer.add(raw_obs, next_raw, action, reward, float(terminated))
            if self.obs_rms is not None:
                self.obs_rms.update(next_raw)

            if terminated or truncated:
                progress.episode(global_step, episode_return)
                episode_return = 0.0
                raw_obs, _ = env.reset()
            else:
                raw_obs = next_raw

            if global_step > c["learning_starts"]:
                if global_step % c["train_frequency"] == 0:
                    i = buffer.sample(int(c["batch_size"]))
                    observations = self._tensor(self._prepare(buffer.observations[i]))
                    next_observations = self._tensor(self._prepare(buffer.next_observations[i]))
                    with torch.no_grad():
                        target_max, _ = self.target_network(next_observations).max(dim=1)
                        td_target = self._tensor(buffer.rewards[i]) + c["gamma"] * target_max * (
                            1 - self._tensor(buffer.dones[i])
                        )
                    actions = torch.as_tensor(buffer.actions[i], device=self.device)[:, None]
                    old_value = self.q_network(observations).gather(1, actions).squeeze(1)
                    loss = F.mse_loss(td_target, old_value)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if global_step % c["target_network_frequency"] == 0:
                    for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_param.data.copy_(c["tau"] * param.data + (1.0 - c["tau"]) * target_param.data)

            progress.tick(global_step + 1)

        progress.tick(total_timesteps, force=True)
        env.close()
        return progress.history

    def _extra_state(self):
        return {"epsilon": self.epsilon}

    def _load_extra_state(self, state):
        self.epsilon = state.get("epsilon", self.epsilon)
