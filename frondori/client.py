"""Client haut niveau : gère la connexion, l'authentification, le protocole
et le décodage des observations, pour ne laisser à l'utilisateur qu'une
seule chose à écrire : une fonction `observation -> action`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import websockets
from gymnasium import spaces

from .errors import AuthenticationError, ConnectionLostError, ProtocolError
from .messages import (
    ActionMessage,
    ActionStatus,
    AuthError,
    Hello,
    MatchEnd,
    MatchStart,
    ObservationMessage,
    Ping,
    Pong,
    Welcome,
    decode_server_message,
    encode_client_message,
)
from .policy import Policy, bind
from .spaces import from_wire, space_from_spec, to_wire

logger = logging.getLogger(__name__)

# Le serveur de compétition Frondori, utilisé quand `url` n'est pas fournie.
# Sous-domaine dédié au serveur de jeu (le site est sur frondori.com) : il
# peut changer de machine sans republier le SDK. Surchargeable sans toucher
# au code par la variable d'environnement FRONDORI_URL (préproduction, serveur
# local...).
DEFAULT_URL = "wss://play.frondori.com/agent"


def default_url() -> str:
    return os.environ.get("FRONDORI_URL", DEFAULT_URL)


@dataclass
class MatchResult:
    match_id: str
    environment: str
    # Nom de l'agent contrôlé pendant ce match (ex: "team_0", "chef_1").
    agent_name: str
    # Somme des récompenses de CHAQUE agent du match.
    returns: dict[str, float]
    # Agents dont le participant s'est déconnecté en cours de match.
    forfeited: list[str]
    # Sort des actions envoyées, tel que rapporté par le serveur. Des
    # `rejected` signalent des actions hors de l'action_space ; des
    # `too_slow`, un calcul plus long que le budget de l'environnement ; des
    # `missing`, des actions jamais arrivées (client bloqué, réseau coupé).
    # Dans tous ces cas, l'action neutre de l'environnement a été jouée.
    actions_applied: int
    actions_rejected: int
    actions_too_slow: int
    actions_missing: int
    # Temps de calcul accordé par action, et temps mesurés pour chacune des
    # actions envoyées (ms) — ceux que le serveur a reçus et enregistrés.
    compute_budget_ms: float
    compute_ms: list[float] = field(default_factory=list)

    @property
    def own_return(self) -> float:
        return self.returns[self.agent_name]

    @property
    def mean_compute_ms(self) -> float | None:
        return sum(self.compute_ms) / len(self.compute_ms) if self.compute_ms else None

    @property
    def max_compute_ms(self) -> float | None:
        return max(self.compute_ms, default=None)


class Agent:
    """
    Point d'entrée du SDK.

    Usage typique (script Python classique) :

        def act(observation):
            ...  # appelle un modèle, retourne une action de l'action_space
            return action

        result = Agent(token="frd_...", environment="kitchen-v0").run(act)
        print(result.own_return, result.returns)

    Sans `url`, l'agent se connecte au serveur de compétition Frondori.
    Avec `local=True`, le match se joue sur ta machine, avec l'environnement
    installé (`pip install frondori-engine frondori-kitchen`), dans les mêmes
    conditions qu'en compétition (cf. `frondori.local`) — sans token :

        result = Agent(environment="kitchen-v0", local=True).run(act)

    En local, les autres agents du match sont joués par ta propre politique
    (self-play), ou par les politiques de `others`, une par autre siège.

    Un même agent (un même token) peut jouer à n'importe quel environnement
    du serveur : c'est `environment` qui choisit. `act` est rappelée une fois par
    pas avec l'observation la plus récente, sous la même forme que dans
    `frondori-engine` en local (dicts de tableaux numpy) : une politique
    entraînée en local se branche telle quelle. Une fois le match démarré,
    `agent.observation_space` et `agent.action_space` (spaces Gymnasium)
    décrivent ce que reçoit et doit renvoyer `act`.

    Depuis un notebook Jupyter (ou tout code déjà dans une boucle asyncio),
    utiliser `await agent.play(act)` à la place de `run(act)`.

    `act` peut aussi être un agent `Policy` (cf. `frondori.policy`) : il
    reçoit alors ses observations dans son format (`observation_format`), et
    s'entraîne en local avec `train` :

        Agent(environment="kitchen-v0", local=True).train(my_agent, total_timesteps=500_000)

    Les matchs se jouent en pas-à-pas : le serveur attend ton action avant
    d'avancer, ta latence réseau ne te coûte donc rien. Ce qui compte, c'est
    le temps de calcul : le SDK mesure le temps entre la réception de
    l'observation et l'envoi de ton action, et l'envoie avec elle. Au-delà du
    budget de l'environnement (`agent.compute_budget_ms`), l'action est
    remplacée par l'action neutre.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        environment: str | None = None,
        client_name: str = "python-sdk",
        *,
        local: bool = False,
        others: list[Callable[[Any], Any]] | None = None,
        seed: int | None = None,
    ) -> None:
        if not environment:
            raise ValueError("`environment` est requis (ex. environment=\"kitchen-v0\")")
        if not local and not token:
            raise ValueError("`token` est requis pour jouer sur un serveur (ou local=True pour jouer en local)")
        if not local and (others is not None or seed is not None):
            raise ValueError("`others` et `seed` ne servent qu'en local (local=True)")
        self.url = None if local else (url or default_url())
        self.token = token
        # Identifiant versionné de l'environnement à jouer (ex: "football-v0").
        self.environment = environment
        self.client_name = client_name
        # Mode local : l'environnement installé au lieu du serveur.
        self.local = local
        # En local : politiques des autres sièges (sinon, self-play), et seed
        # de l'épisode et du tirage du siège (reproductible).
        self.others = others
        self.seed = seed
        # Rempli au `Welcome` (connexion acceptée).
        self.player_id: str | None = None
        # Remplis au `MatchStart` (tout début du match).
        self.match_id: str | None = None
        self.agent_name: str | None = None
        self.observation_space: spaces.Space | None = None
        self.action_space: spaces.Space | None = None
        # Temps de calcul accordé par action, en ms.
        self.compute_budget_ms: float | None = None

    def train(self, policy: Policy, **kwargs) -> Any:
        """Entraîne `policy` sur l'environnement installé, en local : appelle
        `policy.learn(make_env, **kwargs)`, où `make_env()` crée un
        environnement Gymnasium à un seul agent (cf. `frondori.training`).
        Les autres sièges sont joués par `policy` elle-même (self-play), ou
        par `others` ; `seed` rend l'entraînement reproductible.

        `policy.training` vaut `True` pendant l'entraînement, `False` après.
        Renvoie ce que renvoie `learn`.
        """
        if not self.local:
            raise ValueError(
                "l'entraînement se fait en local : "
                f"Agent(environment={self.environment!r}, local=True).train(...)"
            )
        if not isinstance(policy, Policy):
            raise TypeError(
                "train() attend un agent qui hérite de frondori.Policy (avec act et learn), "
                f"reçu {type(policy).__name__}"
            )
        from .training import env_factory

        make_env = env_factory(self.environment, policy, self.others, self.seed)
        policy.training = True
        try:
            return policy.learn(make_env, **kwargs)
        finally:
            policy.training = False

    def run(self, act: Callable[[Any], Any]) -> MatchResult:
        """
        Point d'entrée BLOQUANT, pour un script Python classique : démarre
        sa propre boucle asyncio (`asyncio.run`) et attend la fin du match.

        Lève `RuntimeError` si une boucle asyncio tourne déjà dans le thread
        courant (notebook Jupyter, code déjà `async`...) : `asyncio.run` ne
        peut pas s'imbriquer. Dans ce cas, utiliser `await agent.play(act)`.
        En local, pas de réseau ni de boucle asyncio : `run()` fonctionne
        partout, notebooks compris.
        """
        if self.local:
            from .local import play_local

            return play_local(self, act)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # Aucune boucle en cours : cas normal pour un script synchrone.
        else:
            raise RuntimeError(
                "Agent.run() ne peut pas être appelé depuis une boucle asyncio "
                "déjà en cours (notebook Jupyter, code déjà async...). "
                "Utilise `await agent.play(act)` à la place."
            )
        return asyncio.run(self.play(act))

    async def play(self, act: Callable[[Any], Any]) -> MatchResult:
        """Point d'entrée ASYNC : joue un match complet et retourne son
        résultat. `run()` n'est qu'un raccourci synchrone autour."""
        if self.local:
            # Pas de réseau : le match se joue directement, d'un bloc.
            from .local import play_local

            return play_local(self, act)
        # `max_size=None` : désactive la limite de taille de frame par défaut
        # de la lib `websockets`, pensée pour d'autres usages.
        #
        # PAS de `async with websockets.connect(...) as socket:` ici : ça
        # ferme la connexion CÔTÉ CLIENT en sortie de bloc, y compris après un
        # `return` réussi — or le serveur a déjà fermé SA part juste après
        # avoir envoyé `MatchEnd`. Le client perd alors une course contre une
        # connexion déjà fermée, et l'exception de fermeture remplace le
        # résultat qu'on venait d'obtenir (bug réel, trouvé en faisant jouer
        # deux vrais clients l'un contre l'autre). La fermeture est donc gérée
        # dans un `finally` séparé, qui ne l'ignore qu'une fois le résultat en
        # main.
        socket = await websockets.connect(self.url, max_size=None)
        try:
            await self._handshake(socket)
            result = await self._play_loop(socket, act)
        except websockets.exceptions.ConnectionClosed as exc:
            # Toute fermeture qui arrive JUSQU'ICI (avant d'avoir obtenu un
            # résultat) est anormale.
            raise ConnectionLostError(
                f"connexion perdue avec le serveur avant la fin du match : {exc}"
            ) from exc
        finally:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await socket.close()
        return result

    async def _handshake(self, socket) -> None:
        hello = Hello(token=self.token, environment=self.environment, client_name=self.client_name)
        await socket.send(encode_client_message(hello))
        message = decode_server_message(await socket.recv())
        match message:
            case Welcome(player_id=player_id):
                self.player_id = player_id
            case AuthError(reason=reason):
                raise AuthenticationError(reason)
            case _:
                raise ProtocolError(f"attendu Welcome juste après Hello, reçu {message!r}")

    async def _play_loop(self, socket, act: Callable[[Any], Any]) -> MatchResult:
        # `act` adaptée aux spaces du match (format d'une `Policy`), dès qu'ils
        # sont connus (`MatchStart`).
        bound_act = act
        counts = {status: 0 for status in ActionStatus}
        last_action = None
        compute_times: list[float] = []

        while True:
            raw = await socket.recv()
            # Le chronomètre part dès la réception : décodage, `act` et
            # encodage comptent dans le temps de calcul, le réseau non.
            received_at = time.perf_counter()
            message = decode_server_message(raw)

            # Chaque type de message reçu a exactement une réaction prévue ;
            # tout le reste est une erreur de protocole.
            match message:
                case MatchStart():
                    self.match_id = message.match_id
                    self.agent_name = message.agent
                    self.observation_space = space_from_spec(message.observation_space)
                    self.action_space = space_from_spec(message.action_space)
                    self.compute_budget_ms = message.compute_budget_ms
                    bound_act = bind(act, self.observation_space, self.action_space)
                case Ping(nonce=nonce):
                    await socket.send(encode_client_message(Pong(nonce=nonce)))
                case ObservationMessage():
                    if self.observation_space is None:
                        raise ProtocolError("observation reçue avant MatchStart")
                    counts[message.last_action] += 1
                    self._warn_first(message.last_action, counts, last_action, compute_times)

                    # Épisode fini pour cet agent : le serveur n'attend plus
                    # d'action, seulement que le match se termine.
                    if message.terminated or message.truncated:
                        continue

                    last_action = bound_act(from_wire(self.observation_space, message.observation))
                    wire_action = to_wire(last_action)
                    compute_ms = (time.perf_counter() - received_at) * 1000
                    compute_times.append(compute_ms)
                    try:
                        await socket.send(
                            encode_client_message(
                                ActionMessage(tick=message.tick, action=wire_action, compute_ms=compute_ms)
                            )
                        )
                    except websockets.exceptions.ConnectionClosed:
                        # Le match a pu s'arrêter entre-temps (forfait de
                        # l'adversaire) : le `MatchEnd` est déjà en route,
                        # on continue pour aller le lire.
                        continue
                case MatchEnd():
                    return MatchResult(
                        match_id=self.match_id,
                        environment=self.environment,
                        agent_name=self.agent_name,
                        returns=message.returns,
                        forfeited=message.forfeited,
                        actions_applied=counts[ActionStatus.APPLIED],
                        actions_rejected=counts[ActionStatus.REJECTED],
                        actions_too_slow=counts[ActionStatus.TOO_SLOW],
                        actions_missing=counts[ActionStatus.MISSING],
                        compute_budget_ms=self.compute_budget_ms,
                        compute_ms=compute_times,
                    )
                case _:
                    raise ProtocolError(f"message inattendu pendant le match : {message!r}")

    def _warn_first(self, status: ActionStatus, counts: dict, last_action, compute_times: list[float]) -> None:
        # Une seule fois par match et par problème : les suivants sont
        # comptés dans `MatchResult`, pas répétés à chaque tick.
        if counts[status] != 1:
            return
        if status is ActionStatus.REJECTED:
            logger.warning(
                "action refusée par le serveur : %r n'appartient pas à l'action_space %s. "
                "L'action neutre a été jouée à la place (total dans MatchResult.actions_rejected).",
                last_action,
                self.action_space,
            )
        elif status is ActionStatus.TOO_SLOW:
            logger.warning(
                "action calculée en %.1f ms, au-delà du budget de %s ms de l'environnement : "
                "l'action neutre a été jouée à la place (total dans MatchResult.actions_too_slow).",
                compute_times[-1] if compute_times else float("nan"),
                self.compute_budget_ms,
            )
        elif status is ActionStatus.MISSING:
            logger.warning(
                "action jamais reçue par le serveur (connexion bloquée ou coupée ?) : "
                "l'action neutre a été jouée à la place (total dans MatchResult.actions_missing)."
            )
