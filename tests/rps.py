"""`rps-v0` : pierre-feuille-ciseaux en 100 manches. L'exemple complet de la
documentation (Create an Environment), qui sert ici d'environnement de test :
ce paquet n'en contient aucun."""
from gymnasium import spaces
from pettingzoo import ParallelEnv

from frondori_engine import scene as sc

PASS, ROCK, PAPER, SCISSORS = range(4)
BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}


class RockPaperScissorsEnv(ParallelEnv):
    metadata = {
        "name": "rps_v0",
        "title": "Rock Paper Scissors",
        "description": "Two agents play 100 rounds of rock-paper-scissors.",
        "documentation": "Each round, play `1` rock, `2` paper or `3` scissors (`0` passes and loses the round).",
        "ranking": "elo",            # a duel: ranked by ELO
        "render_modes": ["scene"],
        "render_fps": 10,            # at most 10 rounds per second
        "compute_budget_ms": 50,     # per action
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, rounds=100):
        self.render_mode = render_mode
        self.rounds = rounds
        self.possible_agents = ["player_0", "player_1"]
        self.agents = []
        # Spaces are created once: PettingZoo requires the SAME object on every call.
        # Observation: what the opponent played last round, and the rounds left.
        self._observation_space = spaces.Dict({
            "opponent_last": spaces.Discrete(4),
            "rounds_left": spaces.Discrete(rounds + 1),
        })
        # 0 must be the neutral action: it is played when an agent's action
        # is invalid, too slow or missing.
        self._action_space = spaces.Discrete(4)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.round, self.wins = 0, {agent: 0 for agent in self.agents}
        self.last = {agent: PASS for agent in self.agents}
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        a, b = (int(actions[agent]) for agent in self.possible_agents)
        if a == b:
            reward_a = 0.0
        elif b == PASS or BEATS.get(a) == b:
            reward_a = 1.0
        else:
            reward_a = -1.0
        rewards = {"player_0": reward_a, "player_1": -reward_a}
        for agent, reward in rewards.items():
            self.wins[agent] += reward > 0
        self.last = {"player_0": a, "player_1": b}
        self.round += 1

        done = self.round >= self.rounds
        observations = self._observations()
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: done for agent in self.agents}
        # Numeric infos are shown as match statistics; `score` is shown as the match score.
        infos = {agent: {"score": int(self.wins[agent])} for agent in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observations(self):
        opponent = {"player_0": "player_1", "player_1": "player_0"}
        return {
            agent: {"opponent_last": self.last[opponent[agent]], "rounds_left": self.rounds - self.round}
            for agent in self.possible_agents
        }

    def render(self):
        text = f"{self.wins['player_0']} - {self.wins['player_1']}"
        return sc.scene(10, 4, [sc.text(5, 2, text, size=1.5)], background="#f8fafc")
