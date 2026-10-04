"""Tests du client haut niveau (`Agent`), en isolation du vrai serveur Rust :
un faux serveur WebSocket minimal simule les scénarios à vérifier."""

import asyncio
import logging
import time

import msgpack
import numpy as np
import pytest
import websockets

from frondori import Agent, AuthenticationError, ConnectionLostError

WELCOME = {"Welcome": {"player_id": "test-player", "environment": "demo-v0"}}
MATCH_START = {
    "MatchStart": {
        "match_id": "11111111-2222-3333-4444-555555555555",
        "environment": "demo-v0",
        "agent": "chef_1",
        "agents": ["chef_0", "chef_1"],
        "observation_space": {
            "type": "dict",
            "spaces": {
                "position": {"type": "box", "shape": [2], "dtype": "float32", "low": [-1.0, -1.0], "high": [1.0, 1.0]},
                "time_remaining": {"type": "discrete", "n": 3, "start": 0},
            },
        },
        "action_space": {"type": "discrete", "n": 6, "start": 0},
        "compute_budget_ms": 200.0,
    }
}


def observation(tick, last_action, truncated=False):
    return {
        "Observation": {
            "tick": tick,
            "observation": {"position": [0.5, -0.5], "time_remaining": 2 - tick},
            "last_action": last_action,
            "reward": 1.0 if tick else 0.0,
            "terminated": False,
            "truncated": truncated,
            "info": {},
        }
    }


def pack(message):
    return msgpack.packb(message, use_bin_type=True)


def serve(handler, scenario):
    """Démarre `handler` comme faux serveur, puis exécute `scenario(url)`."""

    async def main():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await scenario(f"ws://127.0.0.1:{port}/agent")

    return asyncio.run(main())


def test_run_raises_when_called_from_a_running_event_loop():
    agent = Agent(url="ws://127.0.0.1:1/agent", token="x", environment="demo-v0")

    async def call_run_from_inside_a_loop():
        agent.run(lambda observation: None)

    with pytest.raises(RuntimeError, match="await agent.play"):
        asyncio.run(call_run_from_inside_a_loop())


def test_authentication_error_is_raised_on_auth_error():
    async def handler(websocket):
        await websocket.recv()
        await websocket.send(pack({"AuthError": {"reason": "unknown token"}}))

    async def scenario(url):
        with pytest.raises(AuthenticationError, match="unknown token"):
            await Agent(url, token="x", environment="demo-v0").play(lambda observation: None)

    serve(handler, scenario)


def test_connection_lost_error_on_unexpected_disconnect():
    async def handler(websocket):
        await websocket.recv()
        await websocket.send(pack(WELCOME))
        await websocket.close(code=1011, reason="crash simulé")

    async def scenario(url):
        with pytest.raises(ConnectionLostError):
            await Agent(url, token="x", environment="demo-v0").play(lambda observation: None)

    serve(handler, scenario)


def test_a_full_match_decodes_observations_and_reports_the_result(caplog):
    hellos = []
    received_actions = []
    late_messages = []

    async def handler(websocket):
        hellos.append(msgpack.unpackb(await websocket.recv(), raw=False))
        await websocket.send(pack(WELCOME))
        await websocket.send(pack(MATCH_START))
        await websocket.send(pack(observation(0, "NotExpected")))
        received_actions.append(msgpack.unpackb(await websocket.recv(), raw=False))
        await websocket.send(pack(observation(1, "Rejected")))
        received_actions.append(msgpack.unpackb(await websocket.recv(), raw=False))
        await websocket.send(pack(observation(2, "TooSlow")))
        received_actions.append(msgpack.unpackb(await websocket.recv(), raw=False))
        # Dernière observation : l'épisode est fini, aucune action attendue.
        await websocket.send(pack(observation(3, "Applied", truncated=True)))
        try:
            late_messages.append(await asyncio.wait_for(websocket.recv(), timeout=0.3))
        except asyncio.TimeoutError:
            pass
        await websocket.send(pack({"MatchEnd": {"returns": {"chef_0": 2.0, "chef_1": 2.0}, "forfeited": []}}))

    seen = []

    def act(observation):
        seen.append(observation)
        time.sleep(0.02)  # un "modèle" qui calcule 20 ms
        return np.int64(4)

    async def scenario(url):
        agent = Agent(url, token="x", environment="demo-v0")
        result = await agent.play(act)
        return agent, result

    with caplog.at_level(logging.WARNING, logger="frondori"):
        agent, result = serve(handler, scenario)

    # L'environnement voulu est demandé par le client, dans le Hello.
    assert hellos[0]["Hello"]["environment"] == "demo-v0"

    # Les observations arrivent décodées selon l'observation_space : même
    # forme qu'en local avec frondori-engine.
    assert seen[0]["position"].dtype == np.float32
    assert seen[0]["position"].tolist() == [0.5, -0.5]
    assert seen[0]["time_remaining"] == 2
    assert agent.observation_space.contains(seen[0])
    assert agent.action_space.n == 6

    # Les actions numpy partent en types natifs, avec le bon tick et le temps
    # de calcul mesuré (au moins les 20 ms de `act`, bien moins qu'1 s).
    assert [m["Action"]["tick"] for m in received_actions] == [0, 1, 2]
    assert all(m["Action"]["action"] == 4 for m in received_actions)
    computes = [m["Action"]["compute_ms"] for m in received_actions]
    assert all(20 <= ms < 1000 for ms in computes)
    assert result.compute_ms == computes
    assert agent.compute_budget_ms == 200.0
    assert late_messages == []

    assert result.agent_name == "chef_1"
    assert result.own_return == 2.0
    assert (result.actions_applied, result.actions_rejected, result.actions_too_slow, result.actions_missing) == (
        1,
        1,
        1,
        0,
    )
    assert result.compute_budget_ms == 200.0
    # L'action refusée et l'action trop lente ont été signalées.
    assert "action rejected" in caplog.text
    assert "over the environment's budget of 200.0 ms" in caplog.text
