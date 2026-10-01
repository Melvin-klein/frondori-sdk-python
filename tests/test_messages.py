"""Tests du module `messages.py`.

Les constantes `*_HEX` ne sont pas arbitraires : ce sont des octets réels
produits par `protocol::encode` côté Rust (serveur `frondori-server`). Si un
de ces tests casse sans qu'on ait touché au SDK, c'est que le format a changé
côté serveur : un vrai changement de version de protocole, pas un bug du SDK.
"""

import msgpack
import pytest

from frondori.errors import ProtocolError
from frondori.messages import (
    ActionMessage,
    ActionStatus,
    Hello,
    MatchEnd,
    ObservationMessage,
    Ping,
    Pong,
    Welcome,
    decode_server_message,
    encode_client_message,
)

WELCOME_HEX = (
    "81a757656c636f6d6582a9706c617965725f6964a9706c617965722d3432ab656e7669726f6e6d"
    "656e74aa6b69746368656e2d7630"
)
MATCH_START_HEX = (
    "81aa4d61746368537461727487a86d617463685f6964d92431313131313131312d323232322d33"
    "3333332d343434342d353535353535353535353535ab656e7669726f6e6d656e74aa6b69746368"
    "656e2d7630a56167656e74a6636865665f31a66167656e747392a6636865665f30a6636865665f"
    "31b16f62736572766174696f6e5f737061636582a474797065a464696374a673706163657382a8"
    "706f736974696f6e85a474797065a3626f78a573686170659102a56474797065a7666c6f617433"
    "32a36c6f7792cbbff0000000000000cbbff0000000000000a46869676892cb3ff0000000000000"
    "cb3ff0000000000000ae74696d655f72656d61696e696e6783a474797065a86469736372657465"
    "a16eccc9a5737461727400ac616374696f6e5f737061636583a474797065a86469736372657465"
    "a16e06a5737461727400b1636f6d707574655f6275646765745f6d73cb4069000000000000"
)
OBSERVATION_HEX = (
    "81ab4f62736572766174696f6e87a47469636b03ab6f62736572766174696f6e82a8706f736974"
    "696f6e92cb3fe0000000000000cbbfd0000000000000ae74696d655f72656d61696e696e67ccc5"
    "ab6c6173745f616374696f6ea852656a6563746564a6726577617264cb3ff0000000000000aa74"
    "65726d696e61746564c2a97472756e6361746564c3a4696e666f81a673657276656401"
)
MATCH_END_HEX = (
    "81a84d61746368456e6482a772657475726e7381a6636865665f30cb4000000000000000a9666f"
    "7266656974656491a6636865665f31"
)
PING_HEX = "81a450696e6781a56e6f6e6365ce0001e240"


def test_decode_welcome():
    assert decode_server_message(bytes.fromhex(WELCOME_HEX)) == Welcome(
        player_id="player-42", environment="kitchen-v0"
    )


def test_decode_match_start_keeps_the_space_descriptions():
    message = decode_server_message(bytes.fromhex(MATCH_START_HEX))
    assert message.compute_budget_ms == 200.0

    assert message.match_id == "11111111-2222-3333-4444-555555555555"
    assert message.agent == "chef_1"
    assert message.agents == ["chef_0", "chef_1"]
    assert message.action_space == {"type": "discrete", "n": 6, "start": 0}
    assert message.observation_space["spaces"]["position"]["shape"] == [2]


def test_decode_observation():
    assert decode_server_message(bytes.fromhex(OBSERVATION_HEX)) == ObservationMessage(
        tick=3,
        observation={"position": [0.5, -0.25], "time_remaining": 197},
        last_action=ActionStatus.REJECTED,
        reward=1.0,
        terminated=False,
        truncated=True,
        info={"served": 1},
    )


def test_decode_match_end():
    assert decode_server_message(bytes.fromhex(MATCH_END_HEX)) == MatchEnd(
        returns={"chef_0": 2.0}, forfeited=["chef_1"]
    )


def test_decode_ping():
    assert decode_server_message(bytes.fromhex(PING_HEX)) == Ping(nonce=123456)


def test_a_malformed_message_is_a_protocol_error():
    raw = msgpack.packb({"Welcome": {"player_id": "sans environnement"}})
    with pytest.raises(ProtocolError, match="Welcome"):
        decode_server_message(raw)


def test_encode_hello():
    raw = encode_client_message(Hello(token="abc", environment="kitchen-v0", client_name="test"))
    assert msgpack.unpackb(raw, raw=False) == {
        "Hello": {"token": "abc", "environment": "kitchen-v0", "client_name": "test"}
    }


def test_encode_pong():
    raw = encode_client_message(Pong(nonce=42))
    assert msgpack.unpackb(raw, raw=False) == {"Pong": {"nonce": 42}}


def test_encode_action_message():
    raw = encode_client_message(ActionMessage(tick=5, action=[[1.0, 0.0], [0.0, 0.5]], compute_ms=12.5))
    assert msgpack.unpackb(raw, raw=False) == {
        "Action": {"tick": 5, "action": [[1.0, 0.0], [0.0, 0.5]], "compute_ms": 12.5}
    }


def test_too_slow_is_a_known_action_status():
    assert ActionStatus("TooSlow") is ActionStatus.TOO_SLOW
