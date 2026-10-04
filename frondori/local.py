"""Match en local : l'environnement installé (`frondori-engine` + le paquet
de l'environnement, ex. `frondori-kitchen`) au lieu du serveur, dans les
mêmes conditions qu'en compétition.

Ce qui est reproduit à l'identique :
- les observations arrivent sous la forme EXACTE du mode en ligne : elles
  passent par le même encodage que sur le réseau (`frondori_engine.wire`
  puis `spaces.from_wire`), types compris ;
- le temps de calcul est mesuré de la même façon (de l'observation reçue à
  l'action prête à partir) et le budget de l'environnement appliqué : au-delà,
  l'action neutre est jouée (`TooSlow`) ;
- une action hors de l'action_space est remplacée par l'action neutre
  (`Rejected`), avec la même validation que le worker du serveur ;
- le résultat est le même `MatchResult`.

Ce qui diffère : pas de réseau, donc jamais d'action `Missing` ; le match
va aussi vite que possible (pas de cadence) ; les autres sièges sont joués
par ta propre politique (self-play) ou par celles passées dans `others`.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from .messages import ActionStatus
from .policy import bind
from .spaces import from_wire, space_from_spec, to_wire

if TYPE_CHECKING:
    from .client import Agent, MatchResult


def play_local(agent: "Agent", act: Callable[[Any], Any]) -> "MatchResult":
    from .client import MatchResult

    engine, wire = import_engine()
    env = make_engine_env(engine, agent.environment)

    seats = list(env.possible_agents)
    others = list(agent.others) if agent.others is not None else None
    if others is not None and len(others) != len(seats) - 1:
        raise ValueError(
            f"{agent.environment} a {len(seats)} agents : `others` doit contenir {len(seats) - 1} "
            f"politique(s), une par autre siège (reçu {len(others)})"
        )

    # Comme en ligne, le siège dépend de l'ordre d'arrivée : tiré au hasard
    # (reproductible avec `seed`). Les autres sièges, dans l'ordre, sont
    # joués par `others`, ou par ta propre politique (self-play).
    rng = random.Random(agent.seed)
    own = rng.choice(seats)
    other_seats = [seat for seat in seats if seat != own]

    budget_ms = float(env.metadata["compute_budget_ms"])
    # Les spaces tels que le SDK les reconstruit en ligne à partir de
    # `MatchStart` : mêmes types, même décodage.
    observation_spaces = {seat: space_from_spec(wire.space_to_spec(env.observation_space(seat))) for seat in seats}
    action_spaces = {seat: space_from_spec(wire.space_to_spec(env.action_space(seat))) for seat in seats}
    # Une `Policy` reçoit ses observations dans son format (cf. `policy.py`),
    # comme en ligne.
    policies = {
        seat: bind(act if seat == own else (others[other_seats.index(seat)] if others else act),
                   observation_spaces[seat], action_spaces[seat])
        for seat in seats
    }
    agent.match_id = f"local-{uuid.uuid4()}"
    agent.agent_name = own
    agent.observation_space = observation_spaces[own]
    agent.action_space = action_spaces[own]
    agent.compute_budget_ms = budget_ms

    counts = {status: 0 for status in ActionStatus}
    compute_times: list[float] = []
    returns = {seat: 0.0 for seat in seats}
    observations, _ = env.reset(seed=agent.seed)

    while env.agents:
        actions = {}
        for seat in env.agents:
            space = env.action_space(seat)
            started = time.perf_counter()
            action = policies[seat](from_wire(observation_spaces[seat], wire.to_wire(observations[seat])))
            value = to_wire(action)
            compute_ms = (time.perf_counter() - started) * 1000

            # Mêmes règles que le serveur et son worker.
            if compute_ms > budget_ms:
                status = ActionStatus.TOO_SLOW
                actions[seat] = wire.neutral_action(space)
            else:
                try:
                    actions[seat] = wire.from_wire(space, value)
                    status = ActionStatus.APPLIED
                except (ValueError, TypeError):
                    actions[seat] = wire.neutral_action(space)
                    status = ActionStatus.REJECTED

            if seat == own:
                compute_times.append(compute_ms)
                counts[status] += 1
                agent._warn_first(status, counts, action, compute_times)

        observations, rewards, *_ = env.step(actions)
        for seat, reward in rewards.items():
            returns[seat] += float(reward)

    return MatchResult(
        match_id=agent.match_id,
        environment=agent.environment,
        agent_name=own,
        returns=returns,
        forfeited=[],
        actions_applied=counts[ActionStatus.APPLIED],
        actions_rejected=counts[ActionStatus.REJECTED],
        actions_too_slow=counts[ActionStatus.TOO_SLOW],
        actions_missing=0,
        compute_budget_ms=budget_ms,
        compute_ms=compute_times,
    )


def make_engine_env(engine, environment: str):
    try:
        return engine.make(environment)
    except KeyError:
        installed = ", ".join(engine.registered_ids()) or "aucun"
        raise ValueError(
            f"environnement {environment!r} non installé (installés : {installed}). "
            f"Installe son paquet, par exemple : pip install frondori-kitchen"
        ) from None


def import_engine():
    try:
        import frondori_engine
        from frondori_engine import wire
    except ImportError:
        raise ImportError(
            "le mode local a besoin de frondori-engine et du paquet de l'environnement : "
            "pip install frondori-engine frondori-kitchen (par exemple)"
        ) from None
    return frondori_engine, wire
