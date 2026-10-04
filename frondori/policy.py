"""La classe de base d'un agent entraînable, et le format de ses observations.

Un agent qui s'entraîne avec le SDK hérite de `Policy` et implémente :
- `act(observation)` : l'action à jouer (à l'entraînement comme en match) ;
- `learn(make_env, **kwargs)` : l'entraînement. `make_env()` crée un
  environnement Gymnasium à un seul agent (cf. `frondori.training`) ; l'agent y
  fait tourner l'algorithme de son choix (CleanRL, Stable-Baselines3, maison).

`observation_format` choisit la forme des observations que reçoit `act`, la
même partout (entraînement, match local, compétition) :
- "dict" (défaut) : telles que l'environnement les produit (dicts de tableaux
  numpy) ;
- "flat" : un seul vecteur float32, ce qu'attendent les réseaux de neurones.
  Les `Discrete`/`MultiDiscrete` y sont encodés en one-hot (comme
  `gymnasium.spaces.flatten`), et une action `Box` y est un vecteur à plat,
  remis à la bonne forme par le SDK.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

FORMATS = ("dict", "flat")


class Policy(abc.ABC):
    """Un agent : une politique (`act`) et sa façon d'apprendre (`learn`).

        class MyAgent(Policy):
            observation_format = "flat"

            def act(self, observation):
                ...

            def learn(self, make_env, total_timesteps=100_000):
                env = make_env()
                ...

        my_agent = MyAgent()
        Agent(environment="kitchen-v0", local=True).train(my_agent, total_timesteps=500_000)
        Agent(environment="kitchen-v0", local=True).run(my_agent)
        Agent(token="frd_…", environment="kitchen-v0").run(my_agent)
    """

    #: Forme des observations reçues par `act` : "dict" ou "flat".
    observation_format: str = "dict"

    #: Vrai pendant `Agent.train()`, faux sinon (comme `model.train()` /
    #: `model.eval()` en PyTorch) : de quoi explorer à l'entraînement
    #: seulement.
    training: bool = False

    @abc.abstractmethod
    def act(self, observation) -> Any:
        """L'action à jouer pour cette observation."""

    @abc.abstractmethod
    def learn(self, make_env: Callable[[], Any], **kwargs) -> Any:
        """Entraîne l'agent. `make_env()` crée un environnement Gymnasium neuf
        (un siège ; les autres joués par cet agent ou par `others`). Les
        `kwargs` sont ceux passés à `Agent.train()`."""

    def __call__(self, observation) -> Any:
        return self.act(observation)


class Formatter:
    """Passage entre les spaces de l'environnement et ceux que voit un agent,
    selon son `observation_format`. Le même objet sert à l'entraînement et en
    match : c'est ce qui garantit qu'`act` reçoit partout la même chose."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, observation_format: str):
        if observation_format not in FORMATS:
            raise ValueError(f"observation_format must be one of {FORMATS}, not {observation_format!r}")
        self.format = observation_format
        self.source_observation_space = observation_space
        self.source_action_space = action_space

        if observation_format == "dict":
            self.observation_space = observation_space
            self.action_space = action_space
            return

        flat = spaces.flatten_space(observation_space)
        self.observation_space = spaces.Box(
            low=flat.low.astype(np.float32), high=flat.high.astype(np.float32), dtype=np.float32
        )
        if isinstance(action_space, spaces.Box):
            self.action_space = spaces.Box(
                low=action_space.low.reshape(-1), high=action_space.high.reshape(-1), dtype=action_space.dtype
            )
        elif isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            self.action_space = action_space
        else:
            raise ValueError(
                f"observation_format=\"flat\" does not handle {type(action_space).__name__} actions "
                "(only Box, Discrete, MultiDiscrete): use observation_format=\"dict\""
            )

    def observation(self, observation):
        if self.format == "dict":
            return observation
        return spaces.flatten(self.source_observation_space, observation).astype(np.float32, copy=False)

    def action(self, action):
        """Action de l'agent -> action de l'environnement (forme d'origine)."""
        if self.format == "flat" and isinstance(self.source_action_space, spaces.Box):
            return np.asarray(action, dtype=self.source_action_space.dtype).reshape(self.source_action_space.shape)
        return action


def is_policy(act) -> bool:
    return isinstance(act, Policy)


def as_policy(act) -> Policy | None:
    """La `Policy` derrière `act` : l'objet lui-même, ou sa méthode `act`
    passée telle quelle (`run(my_agent.act)`) — sans ça, un agent "flat"
    recevrait des dicts en match et des vecteurs à l'entraînement."""
    if isinstance(act, Policy):
        return act
    owner = getattr(act, "__self__", None)
    if isinstance(owner, Policy) and getattr(act, "__func__", None) is type(owner).act:
        return owner
    return None


def bind(act, observation_space: spaces.Space, action_space: spaces.Space) -> Callable[[Any], Any]:
    """`act` (fonction ou `Policy`) -> fonction observation -> action dans les
    spaces de l'environnement, conversion de format comprise."""
    policy = as_policy(act)
    if policy is None or policy.observation_format == "dict":
        return act
    formatter = Formatter(observation_space, action_space, policy.observation_format)
    return lambda observation: formatter.action(policy.act(formatter.observation(observation)))
