"""Le mode local (`Agent(..., local=True)`) : l'environnement installé au lieu
du serveur, dans les mêmes conditions qu'en compétition. Joué ici sur
l'environnement d'exemple de la documentation (`tests/rps.py`, pierre-feuille-
ciseaux), enregistré à la main : ces tests ne dépendent d'aucun paquet
d'environnement installé."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

import frondori_engine
from frondori import ActionStatus, Agent
from frondori.client import DEFAULT_URL

sys.path.insert(0, str(Path(__file__).parent))
from rps import PAPER, ROCK, RockPaperScissorsEnv  # noqa: E402

if "rps-v0" not in frondori_engine.registered_ids():
    frondori_engine.register("rps-v0", RockPaperScissorsEnv)


class SlowRps(RockPaperScissorsEnv):
    metadata = {**RockPaperScissorsEnv.metadata, "compute_budget_ms": 5}


if "slow_rps-v0" not in frondori_engine.registered_ids():
    frondori_engine.register("slow_rps-v0", SlowRps, rounds=3)


def test_a_full_local_match_in_self_play():
    seen = []

    def act(observation):
        seen.append(observation)
        return ROCK

    agent = Agent(environment="rps-v0", local=True, seed=1)
    result = agent.run(act)

    # Self-play : la même politique joue les deux sièges, manches nulles.
    assert result.returns == {"player_0": 0.0, "player_1": 0.0}
    assert result.agent_name in ("player_0", "player_1")
    assert result.match_id.startswith("local-")
    assert (result.actions_applied, result.actions_rejected, result.actions_too_slow, result.actions_missing) == (100, 0, 0, 0)
    assert result.compute_budget_ms == 50.0
    assert len(result.compute_ms) == 100
    # Les deux sièges sont passés par `act` : 200 observations.
    assert len(seen) == 200
    # Même forme qu'en ligne : Discrete -> int Python.
    assert type(seen[0]["opponent_last"]) is int
    assert agent.action_space.n == 4


def test_others_play_the_other_seats():
    result = Agent(environment="rps-v0", local=True, others=[lambda observation: PAPER]).run(lambda observation: ROCK)

    # La pierre perd contre la feuille, 100 fois.
    assert result.own_return == -100.0


def test_the_seat_is_drawn_like_the_order_of_arrival_and_reproducible():
    seats = {Agent(environment="rps-v0", local=True, seed=seed).run(lambda o: ROCK).agent_name for seed in range(20)}
    assert seats == {"player_0", "player_1"}
    first = Agent(environment="rps-v0", local=True, seed=7).run(lambda o: ROCK).agent_name
    assert Agent(environment="rps-v0", local=True, seed=7).run(lambda o: ROCK).agent_name == first


def test_invalid_actions_are_replaced_by_the_neutral_action(caplog):
    result = Agent(environment="rps-v0", local=True, others=[lambda o: ROCK]).run(lambda o: 42)

    assert result.actions_rejected == 100
    # Action neutre (0 : passer) contre la pierre : 100 manches perdues.
    assert result.own_return == -100.0
    assert "action refusée" in caplog.text


def test_the_compute_budget_is_enforced_as_in_competition(caplog):
    def slow(observation):
        time.sleep(0.02)  # 20 ms, budget de 5 ms
        return ROCK

    result = Agent(environment="slow_rps-v0", local=True, others=[lambda o: ROCK]).run(slow)

    assert result.actions_too_slow == 3
    assert result.max_compute_ms >= 20
    assert "au-delà du budget de 5.0 ms" in caplog.text


def test_numpy_actions_are_accepted():
    result = Agent(environment="rps-v0", local=True, others=[lambda o: ROCK]).run(lambda o: np.int64(PAPER))
    assert result.own_return == 100.0


def test_an_environment_that_is_not_installed_says_how_to_install_it():
    with pytest.raises(ValueError, match="non installé.*pip install"):
        Agent(environment="chess-v0", local=True).run(lambda o: 0)


def test_others_must_cover_every_other_seat():
    with pytest.raises(ValueError, match="1 politique"):
        Agent(environment="rps-v0", local=True, others=[lambda o: 0, lambda o: 0]).run(lambda o: 0)


def test_without_url_the_agent_goes_to_the_frondori_server(monkeypatch):
    monkeypatch.delenv("FRONDORI_URL", raising=False)
    # L'adresse publiée avec le SDK : la changer casse tous les clients installés.
    assert DEFAULT_URL == "wss://play.frondori.com/agent"
    assert Agent(token="frd_x", environment="kitchen-v0").url == DEFAULT_URL
    monkeypatch.setenv("FRONDORI_URL", "ws://localhost:8080/agent")
    assert Agent(token="frd_x", environment="kitchen-v0").url == "ws://localhost:8080/agent"
    assert Agent(url="ws://other/agent", token="frd_x", environment="kitchen-v0").url == "ws://other/agent"


def test_a_token_is_required_to_play_on_a_server():
    with pytest.raises(ValueError, match="token"):
        Agent(environment="kitchen-v0")
    with pytest.raises(ValueError, match="local=True"):
        Agent(token="frd_x", environment="kitchen-v0", seed=1)
    assert ActionStatus.TOO_SLOW.value == "TooSlow"
