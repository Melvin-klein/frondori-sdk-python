"""L'entraînement avec le SDK : `Policy`, le format des observations,
`SeatEnv` (un environnement multi-agent vu par un seul agent) et
`Agent.train`. Sans PyTorch : ces tests valent pour un agent écrit "maison"."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import frondori_engine
from frondori import Agent, Policy
from frondori.policy import Formatter, as_policy
from frondori.training import SeatEnv, env_factory

sys.path.insert(0, str(Path(__file__).parent))
from rps import PAPER, ROCK, RockPaperScissorsEnv  # noqa: E402
from target import TargetEnv  # noqa: E402

for env_id, env_class in (("rps-v0", RockPaperScissorsEnv), ("target-v0", TargetEnv)):
    if env_id not in frondori_engine.registered_ids():
        frondori_engine.register(env_id, env_class)


class Recorder(Policy):
    """Un agent qui note ce qu'il voit, et dont `learn` joue quelques pas."""

    def __init__(self, observation_format="dict", action=ROCK):
        self.observation_format = observation_format
        self.action = action
        self.seen = []
        self.learn_calls = []

    def act(self, observation):
        self.seen.append((self.training, observation))
        return self.action

    def learn(self, make_env, steps=3, **kwargs):
        self.learn_calls.append({"training": self.training, "steps": steps, **kwargs})
        env = make_env()
        env.reset(seed=0)
        for _ in range(steps):
            env.step(self.action)
        return "appris"


# -- Policy et formats -----------------------------------------------------------


def test_learn_is_mandatory():
    class OnlyAct(Policy):
        def act(self, observation):
            return 0

    with pytest.raises(TypeError):
        OnlyAct()


def test_the_flat_format_turns_dicts_into_one_float_vector():
    observation_space = spaces.Dict({"opponent_last": spaces.Discrete(4), "position": spaces.Box(-1, 1, (2,))})
    formatter = Formatter(observation_space, spaces.Discrete(4), "flat")
    vector = formatter.observation({"opponent_last": 2, "position": np.array([0.5, -0.5], dtype=np.float32)})
    assert vector.dtype == np.float32
    assert vector.tolist() == [0, 0, 1, 0, 0.5, -0.5]  # Discrete en one-hot
    assert formatter.observation_space.shape == (6,)
    assert formatter.action_space == spaces.Discrete(4)


def test_the_flat_format_flattens_box_actions_and_restores_their_shape():
    formatter = Formatter(spaces.Box(0, 1, (1,)), spaces.Box(-1, 1, (1, 2), np.float32), "flat")
    assert formatter.action_space.shape == (2,)
    assert formatter.action([0.5, -0.5]).shape == (1, 2)


def test_the_dict_format_changes_nothing():
    observation_space = spaces.Dict({"a": spaces.Discrete(2)})
    formatter = Formatter(observation_space, spaces.Discrete(2), "dict")
    assert formatter.observation({"a": 1}) == {"a": 1}
    assert formatter.observation_space is observation_space


def test_a_bound_act_method_is_recognised_as_its_policy():
    agent = Recorder()
    assert as_policy(agent) is agent
    assert as_policy(agent.act) is agent
    assert as_policy(lambda observation: 0) is None


# -- SeatEnv ------------------------------------------------------------------------


@pytest.mark.parametrize("observation_format", ["dict", "flat"])
def test_a_seat_is_a_valid_gymnasium_env(observation_format):
    check_env(SeatEnv("rps-v0", Recorder(observation_format)), skip_render_check=True)


def test_a_seat_env_plays_one_seat_and_the_policy_plays_the_others():
    agent = Recorder("flat", action=PAPER)
    env = SeatEnv("rps-v0", agent)
    observation, info = env.reset(seed=3)
    assert info["seat"] in ("player_0", "player_1")
    assert observation.shape == env.observation_space.shape

    _, reward, terminated, truncated, info = env.step(ROCK)
    # L'autre siège est joué par l'agent (self-play) : PAPER bat ROCK.
    assert reward == -1.0
    assert len(agent.seen) == 1 and agent.seen[0][1].dtype == np.float32
    assert not (terminated or truncated)


def test_others_play_the_other_seats_instead_of_the_policy():
    agent = Recorder()
    env = SeatEnv("rps-v0", agent, others=[lambda observation: ROCK])
    env.reset(seed=0)
    _, reward, *_ = env.step(PAPER)
    assert reward == 1.0
    assert agent.seen == []


def test_an_episode_ends_like_the_match():
    env = SeatEnv("rps-v0", Recorder(), others=[lambda observation: ROCK])
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, terminated, truncated, _ = env.step(PAPER)
        steps, done = steps + 1, terminated or truncated
    assert steps == 100


def test_an_invalid_action_plays_the_neutral_action_and_says_so():
    env = SeatEnv("rps-v0", Recorder(), others=[lambda observation: ROCK])
    env.reset(seed=0)
    _, reward, _, _, info = env.step(42)
    assert info["action_rejected"] is True
    assert reward == -1.0  # passer (l'action neutre) perd la manche


def test_the_seat_is_drawn_at_random_and_reproducibly():
    def seats(seed):
        env = SeatEnv("rps-v0", Recorder())
        env.reset(seed=seed)
        drawn = []
        for _ in range(20):
            drawn.append(env.reset()[1]["seat"])
        return drawn

    assert seats(1) == seats(1)
    assert set(seats(1)) == {"player_0", "player_1"}


def test_continuous_actions_are_given_flat_and_played_in_their_shape():
    env = SeatEnv("target-v0", Recorder("flat", action=np.array([0.5, -0.5], dtype=np.float32)))
    assert env.action_space.shape == (2,)
    env.reset(seed=0)
    _, reward, *_ = env.step(np.array([0.5, -0.5], dtype=np.float32))
    assert reward == pytest.approx(0.0)


def test_env_factory_seeds_each_env_differently():
    make_env = env_factory("rps-v0", Recorder(), seed=7)
    first, second = make_env(), make_env()
    assert first.np_random_seed == 7 and second.np_random_seed == 8


def test_cleanrl_style_vector_envs_work():
    make_env = env_factory("rps-v0", Recorder("flat"), others=[lambda observation: ROCK], seed=0)
    envs = gym.vector.SyncVectorEnv([make_env] * 4)
    observations, _ = envs.reset(seed=0)
    assert observations.shape == (4,) + envs.single_observation_space.shape
    _, rewards, *_ = envs.step(np.full(4, PAPER))
    assert rewards.tolist() == [1.0] * 4


# -- Agent.train ---------------------------------------------------------------------


def test_train_calls_learn_in_training_mode_and_forwards_its_arguments():
    agent = Recorder()
    result = Agent(environment="rps-v0", local=True).train(agent, steps=5, lr=0.1)
    assert result == "appris"
    assert agent.learn_calls == [{"training": True, "steps": 5, "lr": 0.1}]
    assert agent.training is False
    assert all(training for training, _ in agent.seen)


def test_train_leaves_training_mode_even_if_learn_fails():
    class Failing(Recorder):
        def learn(self, make_env, **kwargs):
            raise RuntimeError("boum")

    agent = Failing()
    with pytest.raises(RuntimeError):
        Agent(environment="rps-v0", local=True).train(agent)
    assert agent.training is False


def test_train_is_local_only_and_needs_a_policy():
    with pytest.raises(ValueError, match="local"):
        Agent(token="frd_x", environment="rps-v0").train(Recorder())
    with pytest.raises(TypeError, match="Policy"):
        Agent(environment="rps-v0", local=True).train(lambda observation: 0)


# -- Le même format en match -----------------------------------------------------------


@pytest.mark.parametrize("pass_method", [False, True])
def test_a_flat_policy_gets_flat_observations_in_a_local_match(pass_method):
    agent = Recorder("flat", action=PAPER)
    result = Agent(environment="rps-v0", local=True, others=[lambda observation: ROCK], seed=0).run(
        agent.act if pass_method else agent
    )
    assert result.own_return == 100.0
    assert all(isinstance(observation, np.ndarray) for _, observation in agent.seen)
    assert all(training is False for training, _ in agent.seen)
