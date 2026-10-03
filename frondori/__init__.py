"""SDK Python pour la plateforme de compétition d'IA Frondori.

Usage minimal :

    from frondori import Agent

    def act(observation):
        return policy(observation)  # une action de agent.action_space

    agent = Agent(token="mon-token", environment="kitchen-v0")  # wss://play.frondori.com
    result = agent.run(act)
    print(result.own_return)

Un même agent peut jouer à n'importe quel environnement du serveur. Les
observations arrivent sous la même forme qu'avec `frondori-engine` en local.
"""

from .client import Agent, MatchResult
from .errors import AuthenticationError, ConnectionLostError, ProtocolError
from .messages import ActionStatus

__all__ = [
    "ActionStatus",
    "Agent",
    "AuthenticationError",
    "ConnectionLostError",
    "MatchResult",
    "ProtocolError",
]
