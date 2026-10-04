"""`target-v0` : un environnement de test à actions continues. Deux agents,
20 pas ; chacun reçoit à chaque pas `-|action - cible|` (somme des écarts),
la cible étant fixe : un agent qui apprend joue la cible. L'action a la
forme (1, 2) pour vérifier la remise en forme des actions à plat."""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from frondori_engine import scene as sc

TARGET = np.array([[0.5, -0.5]], dtype=np.float32)


class TargetEnv(ParallelEnv):
    metadata = {
        "name": "target_v0",
        "title": "Target",
        "description": "Play the target.",
        "documentation": "Reward: minus the distance between your action and a fixed target.",
        "ranking": "mean_return",
        "render_modes": ["scene"],
        "render_fps": 10,
        "compute_budget_ms": 50,
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, steps=20):
        self.render_mode = render_mode
        self.steps = steps
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = []
        self._observation_space = spaces.Dict({"steps_left": spaces.Box(0.0, float(steps), (1,), np.float32)})
        self._action_space = spaces.Box(-1.0, 1.0, (1, 2), np.float32)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.step_count = 0
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        rewards = {agent: -float(np.abs(np.asarray(actions[agent]) - TARGET).sum()) for agent in self.agents}
        self.step_count += 1
        done = self.step_count >= self.steps
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations = self._observations()
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, {agent: {} for agent in self.possible_agents}

    def _observations(self):
        left = np.array([self.steps - self.step_count], dtype=np.float32)
        return {agent: {"steps_left": left.copy()} for agent in self.possible_agents}

    def render(self):
        return sc.scene(1, 1, [])
