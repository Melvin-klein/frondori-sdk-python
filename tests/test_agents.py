"""Les agents prêts à l'emploi (`frondori.agents`) apprennent vraiment : sur
des tâches simples, en quelques secondes, ils trouvent la bonne réponse.
Ignorés si PyTorch n'est pas installé (pip install "frondori-sdk[train]")."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import frondori_engine  # noqa: E402
from frondori import Agent  # noqa: E402
from frondori.agents import DQN, PPO, SAC  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from rps import ROCK, RockPaperScissorsEnv  # noqa: E402
from target import TARGET, TargetEnv  # noqa: E402

for env_id, env_class in (("rps-v0", RockPaperScissorsEnv), ("target-v0", TargetEnv)):
    if env_id not in frondori_engine.registered_ids():
        frondori_engine.register(env_id, env_class)


def always_rock(observation):
    return ROCK


def win_rate_against_rock(agent) -> float:
    result = Agent(environment="rps-v0", local=True, others=[always_rock], seed=0).run(agent)
    return (result.own_return + 100) / 200  # retour de -100 (tout perdu) à +100 (tout gagné)


def test_ppo_learns_to_beat_rock():
    agent = PPO(seed=0, verbose=False)
    Agent(environment="rps-v0", local=True, others=[always_rock], seed=0).train(agent, total_timesteps=20_000)
    assert win_rate_against_rock(agent) > 0.9


def test_dqn_learns_to_beat_rock():
    agent = DQN(seed=0, verbose=False, learning_starts=500, target_network_frequency=100)
    Agent(environment="rps-v0", local=True, others=[always_rock], seed=0).train(agent, total_timesteps=6_000)
    assert win_rate_against_rock(agent) > 0.9


def mean_distance_to_target(agent) -> float:
    result = Agent(environment="target-v0", local=True, others=[lambda observation: TARGET], seed=0).run(agent)
    return -result.own_return / 20  # retour = - somme des distances sur 20 pas


def test_sac_learns_to_play_the_target():
    agent = SAC(seed=0, verbose=False, learning_starts=300, batch_size=64)
    Agent(environment="target-v0", local=True, seed=0).train(agent, total_timesteps=2_000)
    assert mean_distance_to_target(agent) < 0.2


def test_ppo_learns_continuous_actions_too():
    # Les réglages de CleanRL visent 1 million de pas, avec un taux
    # d'apprentissage qui décroît jusqu'à 0 : sans décroissance, 30 000 suffisent ici.
    agent = PPO(seed=0, verbose=False, deterministic=True, num_steps=256, num_minibatches=4, anneal_lr=False)
    Agent(environment="target-v0", local=True, seed=0).train(agent, total_timesteps=30_000)
    assert mean_distance_to_target(agent) < 0.3


def test_a_saved_agent_plays_exactly_like_the_original(tmp_path):
    agent = PPO(seed=0, verbose=False, deterministic=True)
    Agent(environment="rps-v0", local=True, others=[always_rock], seed=0).train(agent, total_timesteps=2_000)
    path = tmp_path / "ppo.pt"
    agent.save(str(path))

    loaded = PPO.load(str(path), deterministic=True)
    observations = np.eye(105, dtype=np.float32)[:, : agent.observation_dim]
    assert [agent.act(o) for o in observations] == [loaded.act(o) for o in observations]


def test_training_can_resume():
    agent = DQN(seed=0, verbose=False, learning_starts=100)
    trainer = Agent(environment="rps-v0", local=True, others=[always_rock], seed=0)
    first = trainer.train(agent, total_timesteps=500)
    second = trainer.train(agent, total_timesteps=500)
    assert first and second  # des épisodes joués à chaque fois, même réseau


def test_agents_refuse_the_wrong_kind_of_action():
    with pytest.raises(ValueError, match="Discrete"):
        Agent(environment="target-v0", local=True).train(DQN(verbose=False), total_timesteps=10)
    with pytest.raises(ValueError, match="Box"):
        Agent(environment="rps-v0", local=True).train(SAC(verbose=False), total_timesteps=10)


def test_an_untrained_agent_says_so():
    with pytest.raises(RuntimeError, match="not trained"):
        PPO().act(np.zeros(3))


def test_unknown_settings_are_refused():
    with pytest.raises(TypeError, match="unknown setting"):
        PPO(learning_rat=1e-3)


def test_ppo_self_play_runs():
    # Sans `others` : l'agent joue aussi l'autre siège pendant qu'il apprend.
    agent = PPO(seed=0, verbose=False)
    history = Agent(environment="rps-v0", local=True, seed=0).train(agent, total_timesteps=1_024)
    assert history
