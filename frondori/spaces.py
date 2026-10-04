"""Les spaces d'un environnement, tels que le serveur les décrit dans
`MatchStart`, et le passage entre le fil (listes, nombres, dicts
MessagePack) et les valeurs numpy.

C'est l'autre moitié de `frondori_engine/wire.py` côté serveur : le format
d'une description de space fait partie du protocole. Le but est qu'un agent
reçoive ses observations EXACTEMENT sous la forme que produit l'environnement
en local (`frondori-engine`), pour qu'une politique entraînée en local se
branche telle quelle en compétition.

Spaces supportés : Box, Discrete, MultiDiscrete, et Dict de ces spaces.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .errors import ProtocolError


def space_from_spec(spec: dict) -> spaces.Space:
    kind = spec.get("type")
    if kind == "box":
        dtype = np.dtype(spec["dtype"])
        return spaces.Box(
            low=np.array(spec["low"], dtype=dtype),
            high=np.array(spec["high"], dtype=dtype),
            shape=tuple(spec["shape"]),
            dtype=dtype,
        )
    if kind == "discrete":
        return spaces.Discrete(spec["n"], start=spec["start"])
    if kind == "multi_discrete":
        return spaces.MultiDiscrete(
            np.array(spec["nvec"]), dtype=np.dtype(spec["dtype"]), start=np.array(spec["start"])
        )
    if kind == "dict":
        return spaces.Dict({key: space_from_spec(sub) for key, sub in spec["spaces"].items()})
    raise ProtocolError(f"space unknown to this SDK: {kind!r} (is the SDK up to date?)")


def from_wire(space: spaces.Space, value):
    """Valeur reçue du fil -> valeur du space (tableaux numpy aux bons
    dtype/forme, entiers pour les `Discrete`, dicts pour les `Dict`)."""
    if isinstance(space, spaces.Dict):
        return {key: from_wire(sub, value[key]) for key, sub in space.spaces.items()}
    if isinstance(space, spaces.Discrete):
        return int(value)
    return np.asarray(value, dtype=space.dtype).reshape(space.shape)


def to_wire(value):
    """numpy -> types natifs MessagePack, récursivement (pour les actions)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(item) for item in value]
    return value
