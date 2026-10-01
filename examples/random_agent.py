"""Exemple : un agent qui joue des actions aléatoires, dans n'importe quel
environnement — le même code pour tous.

Usage : python examples/random_agent.py <token> <environnement> [url]
        (ex. python examples/random_agent.py frd_... kitchen-v0)
"""

import sys

from frondori import Agent


def main() -> None:
    token, environment = sys.argv[1], sys.argv[2]
    url = sys.argv[3] if len(sys.argv) > 3 else "ws://127.0.0.1:8080/agent"
    agent = Agent(url=url, token=token, environment=environment, client_name="random-agent-python")

    # `agent.action_space` est connu dès le début du match (MatchStart),
    # c'est-à-dire avant le premier appel à `act`.
    result = agent.run(lambda observation: agent.action_space.sample())

    print(f"{result.environment} : j'étais {result.agent_name}, retour {result.own_return}")
    print(f"retours de tous les agents : {result.returns}")


if __name__ == "__main__":
    main()
