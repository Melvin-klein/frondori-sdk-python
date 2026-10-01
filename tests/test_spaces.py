"""Les descriptions de spaces ci-dessous sont celles que le serveur envoie
réellement pour `kitchen-v0` et `football-v0` (sortie du worker
`frondori_engine.worker`, `cmd: describe`), réduites à ce qui sert ici."""

import numpy as np
import pytest
from gymnasium import spaces

from frondori.errors import ProtocolError
from frondori.spaces import from_wire, space_from_spec, to_wire

INF = float("inf")

KITCHEN_OBSERVATION = {
    "type": "dict",
    "spaces": {
        "self": {"type": "multi_discrete", "nvec": [4, 5, 4, 4], "start": [0, 0, 0, 0], "dtype": "int64"},
        "time_remaining": {"type": "discrete", "n": 201, "start": 0},
    },
}
FOOTBALL_ACTION = {
    "type": "box",
    "shape": [3, 5],
    "dtype": "float32",
    "low": [[-1.0, -1.0, -1.0, -1.0, 0.0]] * 3,
    "high": [[1.0] * 5] * 3,
}
FOOTBALL_BALL = {"type": "box", "shape": [4], "dtype": "float32", "low": [-INF] * 4, "high": [INF] * 4}


def test_kitchen_observation_is_decoded_like_the_local_environment():
    space = space_from_spec(KITCHEN_OBSERVATION)

    observation = from_wire(space, {"self": [1, 1, 0, 0], "time_remaining": 200})

    assert isinstance(space, spaces.Dict)
    assert observation["self"].dtype == np.int64
    assert observation["self"].tolist() == [1, 1, 0, 0]
    assert observation["time_remaining"] == 200
    assert space.contains(observation)


def test_football_action_space_is_rebuilt_with_its_bounds():
    space = space_from_spec(FOOTBALL_ACTION)

    assert space.shape == (3, 5)
    assert space.dtype == np.float32
    assert space.contains(np.zeros((3, 5), dtype=np.float32))
    assert not space.contains(np.full((3, 5), 2.0, dtype=np.float32))


def test_infinite_bounds_survive():
    space = space_from_spec(FOOTBALL_BALL)
    assert np.isinf(space.high).all()
    assert from_wire(space, [0.1, 0.2, 0.3, 0.4]).shape == (4,)


def test_actions_are_turned_into_plain_types_for_the_wire():
    assert to_wire(np.zeros((2, 2), dtype=np.float32)) == [[0.0, 0.0], [0.0, 0.0]]
    assert to_wire(np.int64(3)) == 3
    assert to_wire({"a": np.array([1, 2])}) == {"a": [1, 2]}


def test_an_unknown_space_type_is_a_protocol_error():
    with pytest.raises(ProtocolError, match="graph"):
        space_from_spec({"type": "graph"})
