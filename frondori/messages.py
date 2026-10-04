"""Messages échangés avec le serveur, et leur (dé)sérialisation MessagePack.

Équivalent Python du crate `protocol` côté Rust. Le format exact sur le fil
est fixé par `protocol::encode`/`decode` (voir `protocol/src/lib.rs`) :

- Chaque message est une map MessagePack à une seule clé,
  `{"NomDuVariant": contenu}` (représentation "externally tagged" de serde
  pour un enum Rust à données). Ici, un dict Python à une seule entrée.
- Les structs Rust sont des maps `{"champ": valeur}` : chaque champ se lit
  par son nom, sans dépendre de l'ordre de déclaration côté Rust.
- Un enum Rust SANS données (`ActionStatus`) est la chaîne du nom de la
  variante (`"Rejected"`, pas `{"Rejected": null}`).

Observations, actions, spaces et infos sont des valeurs dont la forme dépend
de l'environnement : ce module les transmet brutes, c'est `spaces.py` qui
les interprète. Les vecteurs de `tests/test_messages.py` sont des octets
réels produits par `protocol::encode` côté Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import msgpack

from .errors import ProtocolError


class ActionStatus(str, Enum):
    """Sort de l'action envoyée pour le tick précédent. Hérite de `str` :
    se construit directement depuis la valeur reçue (`ActionStatus("Rejected")`)."""

    NOT_EXPECTED = "NotExpected"
    APPLIED = "Applied"
    # Invalide pour l'action_space : l'action neutre a été appliquée à la place.
    REJECTED = "Rejected"
    # Calculée en plus que le budget de l'environnement : l'action neutre a
    # été appliquée à la place.
    TOO_SLOW = "TooSlow"
    # Jamais reçue (délai réseau du serveur dépassé) : l'action neutre a été
    # appliquée.
    MISSING = "Missing"


@dataclass
class Hello:
    token: str
    environment: str
    client_name: str


@dataclass
class Welcome:
    player_id: str
    environment: str


@dataclass
class AuthError:
    reason: str


@dataclass
class MatchStart:
    match_id: str
    environment: str
    agent: str
    agents: list[str]
    observation_space: dict
    action_space: dict
    # Temps de calcul accordé pour chaque action, en millisecondes.
    compute_budget_ms: float


@dataclass
class ActionMessage:
    tick: int
    action: Any
    # Temps mis à produire l'action (de la réception de l'observation à
    # l'envoi de l'action), en millisecondes.
    compute_ms: float


@dataclass
class ObservationMessage:
    tick: int
    observation: Any
    last_action: ActionStatus
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


@dataclass
class MatchEnd:
    returns: dict[str, float]
    forfeited: list[str]


@dataclass
class Ping:
    nonce: int


@dataclass
class Pong:
    nonce: int


ServerMessage = Welcome | AuthError | MatchStart | ObservationMessage | MatchEnd | Ping


def decode_server_message(raw: bytes) -> ServerMessage:
    """Décode un message brut reçu du serveur (frame WebSocket binaire)."""
    data = msgpack.unpackb(raw, raw=False)
    if not isinstance(data, dict) or len(data) != 1:
        raise ProtocolError(f"invalid message envelope: {data!r}")
    ((variant, payload),) = data.items()

    try:
        match variant:
            case "Welcome":
                return Welcome(player_id=payload["player_id"], environment=payload["environment"])
            case "AuthError":
                return AuthError(reason=payload["reason"])
            case "MatchStart":
                return MatchStart(
                    match_id=payload["match_id"],
                    environment=payload["environment"],
                    agent=payload["agent"],
                    agents=list(payload["agents"]),
                    observation_space=payload["observation_space"],
                    action_space=payload["action_space"],
                    compute_budget_ms=payload["compute_budget_ms"],
                )
            case "Observation":
                return ObservationMessage(
                    tick=payload["tick"],
                    observation=payload["observation"],
                    last_action=ActionStatus(payload["last_action"]),
                    reward=payload["reward"],
                    terminated=payload["terminated"],
                    truncated=payload["truncated"],
                    info=payload["info"] or {},
                )
            case "MatchEnd":
                return MatchEnd(returns=dict(payload["returns"]), forfeited=list(payload["forfeited"]))
            case "Ping":
                return Ping(nonce=payload["nonce"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed {variant!r} message: {exc!r}") from exc
    raise ProtocolError(f"unknown server message: {variant!r} (is the SDK up to date?)")


def encode_client_message(message: Hello | ActionMessage | Pong) -> bytes:
    """Encode un message à envoyer au serveur (frame WebSocket binaire).
    `ActionMessage.action` doit déjà être sous sa forme "fil" (cf.
    `spaces.to_wire`)."""
    if isinstance(message, Hello):
        payload = {
            "Hello": {"token": message.token, "environment": message.environment, "client_name": message.client_name}
        }
    elif isinstance(message, ActionMessage):
        payload = {"Action": {"tick": message.tick, "action": message.action, "compute_ms": message.compute_ms}}
    elif isinstance(message, Pong):
        payload = {"Pong": {"nonce": message.nonce}}
    else:
        raise ProtocolError(f"unknown client message type: {message!r}")
    return msgpack.packb(payload, use_bin_type=True)
