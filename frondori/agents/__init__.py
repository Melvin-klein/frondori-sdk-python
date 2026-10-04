"""Agents prêts à l'emploi, adaptés de CleanRL (https://github.com/vwxyzjn/cleanrl) :

- `PPO` : actions `Discrete` ou `Box` ;
- `DQN` : actions `Discrete` ;
- `SAC` : actions `Box` bornées.

Ils ont besoin de PyTorch : pip install "frondori-sdk[train]".

    from frondori import Agent
    from frondori.agents import PPO

    agent = PPO()
    Agent(environment="kitchen-v0", local=True).train(agent, total_timesteps=500_000)
    agent.save("ppo.pt")

    agent = PPO.load("ppo.pt")
    Agent(token="frd_…", environment="kitchen-v0").run(agent)

Chacun tient dans un fichier lisible, pour être copié et modifié : c'est
aussi un exemple de `frondori.Policy`. Licences : THIRD_PARTY_LICENSES.md.
"""

from .dqn import DQN
from .ppo import PPO
from .sac import SAC

__all__ = ["DQN", "PPO", "SAC"]
