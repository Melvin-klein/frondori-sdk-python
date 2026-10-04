"""Briques communes aux agents prêts à l'emploi : PyTorch, normalisation des
observations, sauvegarde, suivi de l'entraînement."""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from ..policy import Policy

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - dépend de l'installation
    raise ImportError(
        'the agents of frondori.agents need PyTorch: pip install "frondori-sdk[train]"'
    ) from exc


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialisation orthogonale, comme CleanRL."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(sizes: list[int], activation: type[nn.Module], last_std: float | None = None) -> nn.Sequential:
    """Perceptron : `sizes` = [entrée, cachées..., sortie]. `last_std` :
    initialisation orthogonale (PPO) ; `None` : initialisation PyTorch."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        if last_std is not None:
            linear = layer_init(linear, std=last_std if i == len(sizes) - 2 else np.sqrt(2))
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class RunningMeanStd:
    """Moyenne et variance glissantes (méthode parallèle de Chan et al.),
    comme `gymnasium.wrappers.NormalizeObservation`. Mises à jour pendant
    l'entraînement, figées en match : elles font partie du modèle."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape((-1,) + self.mean.shape)
        batch_mean, batch_var, batch_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    @classmethod
    def from_state(cls, state: dict) -> "RunningMeanStd":
        rms = cls()
        rms.mean = np.asarray(state["mean"], dtype=np.float64)
        rms.var = np.asarray(state["var"], dtype=np.float64)
        rms.count = state["count"]
        return rms


def space_state(space: spaces.Space) -> dict:
    """Un space d'action en types simples (pour `torch.load(weights_only=True)`)."""
    if isinstance(space, spaces.Discrete):
        return {"type": "discrete", "n": int(space.n)}
    if isinstance(space, spaces.Box):
        return {"type": "box", "low": space.low.tolist(), "high": space.high.tolist(), "dtype": str(space.dtype)}
    raise ValueError(f"unsupported space: {space}")


def space_from_state(state: dict) -> spaces.Space:
    if state["type"] == "discrete":
        return spaces.Discrete(state["n"])
    dtype = np.dtype(state["dtype"])
    return spaces.Box(low=np.asarray(state["low"], dtype=dtype), high=np.asarray(state["high"], dtype=dtype), dtype=dtype)


class Progress:
    """Retours par épisode (la vraie récompense de l'environnement : ni
    normalisée, ni façonnée par un wrapper) et affichage régulier de
    l'avancement."""

    def __init__(self, total_timesteps: int, verbose: bool, every_seconds: float = 10.0):
        self.total = total_timesteps
        self.verbose = verbose
        self.every = every_seconds
        self.started = time.time()
        self.last_print = self.started
        self.recent: deque[float] = deque(maxlen=20)
        self.history: list[dict] = []

    def episode(self, step: int, episode_return: float) -> None:
        self.recent.append(episode_return)
        self.history.append({"step": step, "return": episode_return})

    def tick(self, step: int, force: bool = False) -> None:
        now = time.time()
        if not self.verbose or (not force and now - self.last_print < self.every):
            return
        self.last_print = now
        mean = f"{np.mean(self.recent):.2f}" if self.recent else "-"
        speed = step / max(now - self.started, 1e-9)
        print(
            f"step {step:>9,} / {self.total:,}  ·  mean return (last {len(self.recent)} episodes) {mean}"
            f"  ·  {speed:,.0f} steps/s",
            flush=True,
        )


class TorchPolicy(Policy):
    """Base des agents PyTorch : observations à plat, normalisation
    facultative, sauvegarde et chargement.

    Les sous-classes déclarent `DEFAULTS` (réglages de CleanRL), construisent
    leurs réseaux dans `_build` et listent les modules à sauvegarder dans
    `_modules`."""

    observation_format = "flat"
    DEFAULTS: dict[str, Any] = {}

    def __init__(
        self,
        *,
        wrap_env: Callable[[Any], Any] | None = None,
        device: str = "cpu",
        seed: int | None = None,
        verbose: bool = True,
        **config: Any,
    ):
        unknown = set(config) - set(self.DEFAULTS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown setting(s) {sorted(unknown)}")
        # Seuls les réglages explicitement donnés : les autres sont choisis à
        # l'entraînement (certains dépendent du type d'action).
        self.overrides = {key: value for key, value in config.items() if value is not None}
        self.config: dict[str, Any] = {}
        self.wrap_env = wrap_env
        self.device = torch.device(device)
        self.seed = seed
        self.verbose = verbose
        self.observation_dim: int | None = None
        self.agent_action_space: spaces.Space | None = None
        self.obs_rms: RunningMeanStd | None = None

    # -- à fournir par les sous-classes ---------------------------------
    def _defaults_for(self, action_space: spaces.Space) -> dict[str, Any]:
        return dict(self.DEFAULTS)

    def _check_action_space(self, action_space: spaces.Space) -> None:
        pass

    def _build(self) -> None:
        raise NotImplementedError

    def _modules(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    # -- commun ---------------------------------------------------------------
    def _make_envs(self, make_env: Callable[[], Any], count: int) -> list:
        envs = []
        for _ in range(count):
            env = make_env()
            envs.append(self.wrap_env(env) if self.wrap_env is not None else env)
        return envs

    def _setup(self, env) -> None:
        """Réseaux construits au premier entraînement (les tailles viennent
        de l'environnement) ; un second `train()` reprend où on en était."""
        if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
            raise ValueError(f"{type(self).__name__} expects flat observations (1D Box), got {env.observation_space}")
        observation_dim = int(env.observation_space.shape[0])
        if self.observation_dim is not None:
            if observation_dim != self.observation_dim or space_state(env.action_space) != space_state(self.agent_action_space):
                raise ValueError("this agent was trained on another environment (different spaces)")
            return
        self._check_action_space(env.action_space)
        self.observation_dim = observation_dim
        self.agent_action_space = env.action_space
        self.config = {**self._defaults_for(env.action_space), **self.overrides}
        if self.config.get("normalize_observations"):
            self.obs_rms = RunningMeanStd((observation_dim,))
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self._build()

    def _prepare(self, observations, update: bool = False) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if self.obs_rms is None:
            return observations
        if update:
            self.obs_rms.update(observations)
        return self.obs_rms.normalize(observations)

    def _tensor(self, array) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self.device)

    def _require_trained(self) -> None:
        if self.observation_dim is None:
            raise RuntimeError(
                f"{type(self).__name__} is not trained: Agent(..., local=True).train(agent), "
                f"or {type(self).__name__}.load(path)"
            )

    def save(self, path: str) -> None:
        """Enregistre l'agent (réseaux, normalisation, réglages)."""
        self._require_trained()
        torch.save(
            {
                "agent": type(self).__name__,
                "config": self.config,
                "observation_dim": self.observation_dim,
                "action_space": space_state(self.agent_action_space),
                "obs_rms": self.obs_rms.state() if self.obs_rms is not None else None,
                "modules": {name: module.state_dict() for name, module in self._modules().items()},
                "extra": self._extra_state(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "TorchPolicy":
        """Recharge un agent enregistré par `save`, prêt à jouer (ou à
        reprendre l'entraînement)."""
        # weights_only : rien que des tenseurs et des types simples, aucun
        # code exécuté au chargement.
        data = torch.load(path, map_location=device, weights_only=True)
        if data["agent"] != cls.__name__:
            raise ValueError(f"{path} holds a {data['agent']} agent, not {cls.__name__}")
        agent = cls(device=device, **kwargs)
        agent.config = dict(data["config"])
        agent.overrides = dict(data["config"])
        agent.observation_dim = data["observation_dim"]
        agent.agent_action_space = space_from_state(data["action_space"])
        agent.obs_rms = RunningMeanStd.from_state(data["obs_rms"]) if data["obs_rms"] is not None else None
        agent._build()
        for name, module in agent._modules().items():
            module.load_state_dict(data["modules"][name])
        agent._load_extra_state(data.get("extra") or {})
        return agent

    def _extra_state(self) -> dict:
        return {}

    def _load_extra_state(self, state: dict) -> None:
        pass
