"""L'entraînement avec le SDK : un environnement Frondori vu par UN agent.

Les environnements Frondori sont multi-agents (PettingZoo) ; les algorithmes
courants (CleanRL, Stable-Baselines3...) attendent un environnement Gymnasium
à un seul agent. `SeatEnv` fait le pont :
- un siège, tiré au hasard à chaque épisode (comme en compétition, où il
  dépend de l'ordre d'arrivée) ;
- les autres sièges joués par l'agent lui-même (self-play), ou par les
  politiques de `others` ;
- les observations sous la forme EXACTE de la compétition (même encodage que
  sur le réseau), au format de l'agent (`observation_format`) ;
- une action invalide remplacée par l'action neutre, comme en match
  (`info["action_rejected"]`).

Pas de budget de calcul à l'entraînement : il ne s'applique qu'en match.

    make_env = training.env_factory("kitchen-v0", my_agent)
    env = make_env()                      # un gymnasium.Env
    observation, info = env.reset(seed=0)
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym

from .local import import_engine, make_engine_env
from .policy import Formatter, as_policy, bind
from .spaces import from_wire, space_from_spec, to_wire


class SeatEnv(gym.Env):
    """Un siège d'un environnement Frondori, en environnement Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, environment: str, policy: Any, others: list | None = None):
        engine, self._wire = import_engine()
        self._env = make_engine_env(engine, environment)
        self.environment = environment
        self._seats = list(self._env.possible_agents)

        if others is not None and len(others) != len(self._seats) - 1:
            raise ValueError(
                f"{environment} has {len(self._seats)} agents: `others` must hold "
                f"{len(self._seats) - 1} policy(ies), one per other seat (got {len(others)})"
            )
        self._learner = policy
        self._others = list(others) if others is not None else None

        # Les spaces tels que le SDK les reconstruit en match (mêmes types).
        spec = self._wire.space_to_spec
        self._observation_spaces = {s: space_from_spec(spec(self._env.observation_space(s))) for s in self._seats}
        self._action_spaces = {s: space_from_spec(spec(self._env.action_space(s))) for s in self._seats}
        first = self._seats[0]
        if any(
            spec(self._env.observation_space(s)) != spec(self._env.observation_space(first))
            or spec(self._env.action_space(s)) != spec(self._env.action_space(first))
            for s in self._seats
        ):
            raise ValueError(
                f"{environment}: its agents do not all share the same spaces; "
                "the SDK cannot (yet) train one agent on different seats: train with frondori_engine instead"
            )

        learner = as_policy(policy)
        observation_format = learner.observation_format if learner is not None else "dict"
        self._formatter = Formatter(self._observation_spaces[first], self._action_spaces[first], observation_format)
        self.observation_space = self._formatter.observation_space
        self.action_space = self._formatter.action_space

        self._own: str | None = None
        self._observations: dict = {}
        self._players: dict[str, Callable[[Any], Any]] = {}

    @property
    def seat(self) -> str | None:
        """Le siège joué par l'agent dans l'épisode en cours."""
        return self._own

    @property
    def raw_observation(self):
        """La dernière observation de l'agent sous sa forme d'origine (dict),
        quel que soit son format : de quoi façonner la récompense dans un
        wrapper Gymnasium (`env.unwrapped.raw_observation`)."""
        return from_wire(self._observation_spaces[self._own], self._wire.to_wire(self._observations[self._own]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._own = self._seats[int(self.np_random.integers(len(self._seats)))]
        others = [s for s in self._seats if s != self._own]
        self._players = {
            seat: bind(
                self._others[i] if self._others is not None else self._learner,
                self._observation_spaces[seat],
                self._action_spaces[seat],
            )
            for i, seat in enumerate(others)
        }
        self._observations, infos = self._env.reset(seed=int(self.np_random.integers(2**31 - 1)))
        return self._own_observation(), {"seat": self._own, **infos.get(self._own, {})}

    def step(self, action):
        if self._own is None:
            raise RuntimeError("step() called before reset()")
        actions, rejected = {}, False
        for seat in self._env.agents:
            if seat == self._own:
                actions[seat], rejected = self._validate(seat, self._formatter.action(action))
            else:
                observation = from_wire(self._observation_spaces[seat], self._wire.to_wire(self._observations[seat]))
                actions[seat], _ = self._validate(seat, self._players[seat](observation))

        self._observations, rewards, terminations, truncations, infos = self._env.step(actions)
        terminated = bool(terminations.get(self._own, False))
        truncated = bool(truncations.get(self._own, False))
        # Épisode terminé pour tout le monde (ou siège retiré) sans drapeau
        # explicite : c'est une fin de partie.
        if self._own not in self._env.agents and not (terminated or truncated):
            terminated = True
        reward = float(rewards.get(self._own, 0.0))
        # `env_reward` : la vraie récompense, même si un wrapper façonne celle
        # que voit l'algorithme (les agents l'utilisent pour leurs bilans).
        info = {"seat": self._own, "action_rejected": rejected, "env_reward": reward, **infos.get(self._own, {})}
        return self._own_observation(), reward, terminated, truncated, info

    def _own_observation(self):
        return self._formatter.observation(self.raw_observation)

    def _validate(self, seat: str, action) -> tuple[Any, bool]:
        """Même règle qu'en match : hors de l'action_space -> action neutre."""
        space = self._env.action_space(seat)
        try:
            return self._wire.from_wire(space, to_wire(action)), False
        except (ValueError, TypeError):
            return self._wire.neutral_action(space), True


def env_factory(environment: str, policy: Any, others: list | None = None, seed: int | None = None) -> Callable[[], SeatEnv]:
    """La fabrique `make_env` passée à `Policy.learn`. Avec `seed`, chaque
    environnement créé reçoit sa propre graine (seed, seed + 1, ...) :
    l'entraînement est reproductible, et des environnements parallèles ne
    jouent pas tous la même partie."""
    created = 0

    def make_env() -> SeatEnv:
        nonlocal created
        env = SeatEnv(environment, policy, others)
        if seed is not None:
            env.reset(seed=seed + created)
        created += 1
        return env

    return make_env

