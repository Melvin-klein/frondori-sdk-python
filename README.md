# frondori-sdk

SDK Python pour connecter un modèle à la plateforme de compétition Frondori.
Il gère la connexion WebSocket, l'authentification et le protocole réseau :
il ne reste qu'à écrire une fonction `observation -> action`.

Le SDK ne connaît aucun jeu en particulier : tu choisis l'environnement à
jouer (football, cuisine coopérative...) à chaque connexion, et le serveur
décrit ses observations et ses actions au début de chaque match. Un même
agent (un même token) peut jouer à plusieurs environnements ; son classement
est tenu séparément pour chacun.

Pour s'entraîner en local, sans serveur, utiliser `frondori-engine` et les
paquets des environnements voulus (`frondori-kitchen`, `frondori-football`...) :
ce sont les mêmes environnements, et **les observations reçues en compétition
ont exactement la même forme qu'en local**. Une politique entraînée en local
se branche donc telle quelle ici — et le SDK sait aussi jouer un match complet
en local (`local=True`).

## Installation

```bash
pip install frondori-sdk

# Pour jouer aussi en local : frondori-engine et les environnements voulus
pip install "frondori-sdk[local]" frondori-kitchen

# Pour entraîner avec les agents prêts à l'emploi (PPO, DQN, SAC) : PyTorch en plus
pip install "frondori-sdk[train]" frondori-kitchen
```

Dépendances : Python >= 3.10, `websockets`, `msgpack`, `numpy`, `gymnasium`.

## Démarrage rapide

```python
from frondori import Agent

agent = Agent(token="frd_…", environment="kitchen-v0")   # sans url : le serveur Frondori

def act(observation):
    return my_policy(observation)  # une action de agent.action_space

result = agent.run(act)
print(result.own_return, result.returns)
```

`act` est rappelée une fois par pas avec l'observation la plus récente ;
tout le reste (handshake, `Ping`, boucle réseau) est géré en interne. Dès le
début du match, `agent.observation_space` et `agent.action_space` (spaces
Gymnasium) décrivent ce que reçoit et doit renvoyer `act`. Voir
[`examples/random_agent.py`](examples/random_agent.py).

### Depuis un notebook Jupyter (ou du code déjà `async`)

`Agent.run()` démarre sa propre boucle asyncio, ce qui échoue si une boucle
tourne déjà. Utiliser `play()` à la place :

```python
result = await agent.play(act)
```

## Jouer en local : évaluer une politique entraînée

Trois étapes, trois outils :

| Étape | Outil | Pour |
|---|---|---|
| 1. Entraîner | `Agent(environment="kitchen-v0", local=True).train(agent)` | Apprendre vite : un agent prêt à l'emploi, ou ton algorithme (CleanRL, Stable-Baselines3) |
| 1 bis. Entraîner | `frondori_engine.make("kitchen-v0")` (API PettingZoo) | Contrôle total : chaque pas, chaque récompense, tous les agents |
| 2. Évaluer | `Agent(environment="kitchen-v0", local=True)` | Vérifier une politique entraînée en conditions de compétition |
| 3. Concourir | `Agent(token="frd_…", environment="kitchen-v0")` | Jouer contre les autres participants, entrer au classement |

`run` ne sert pas à entraîner : `act` ne reçoit que l'observation, jamais
la récompense, et le résultat n'arrive qu'en fin de match (pour entraîner :
`train`, ci-dessous). Pour les étapes 2 et 3, le code est le même : seuls les
paramètres d'`Agent` changent.

```python
# Sur le serveur Frondori (wss://play.frondori.com/agent par défaut, surchargeable par FRONDORI_URL)
agent = Agent(token="frd_…", environment="kitchen-v0")

# En local, avec l'environnement installé : pas de token
agent = Agent(environment="kitchen-v0", local=True)

result = agent.run(act)
```

Le match local reproduit les conditions de la compétition : mêmes
observations (mêmes types), budget de calcul appliqué (au-delà : action
neutre, `actions_too_slow`), actions invalides remplacées, même
`MatchResult`. Ton siège est tiré au hasard, comme l'ordre d'arrivée en
ligne. Pas de réseau, donc jamais d'action manquante, et le match va aussi
vite que tes politiques.

Les autres sièges sont joués par ta propre politique (self-play), ou par
celles que tu fournis dans `others`, une par autre siège :

```python
result = Agent(environment="football-v0", local=True, others=[baseline.act], seed=0).run(act)
```

`seed` rend l'épisode et le tirage du siège reproductibles. En self-play, le
même appelable joue tous les sièges : si ta politique garde un état, donne
aux autres sièges leurs propres instances via `others`. En local, `run()`
fonctionne partout, notebooks compris.

## Entraîner avec le SDK

Un agent entraînable hérite de `frondori.Policy` et implémente deux méthodes :
`act(observation)` (jouer) et `learn(make_env, **kwargs)` (apprendre). `train`
l'entraîne en local, puis le même objet joue en match :

```python
from frondori import Agent
from frondori.agents import PPO     # pip install "frondori-sdk[train]"

agent = PPO()
Agent(environment="kitchen-v0", local=True).train(agent, total_timesteps=500_000)
agent.save("ppo.pt")

Agent(environment="kitchen-v0", local=True).run(agent)             # évaluer
Agent(token="frd_…", environment="kitchen-v0").run(PPO.load("ppo.pt"))  # concourir
```

`train(agent, **kwargs)` met `agent.training` à `True` et appelle
`agent.learn(make_env, **kwargs)`. Chaque `make_env()` crée un environnement
**Gymnasium à un seul agent** : le format qu'attendent CleanRL,
Stable-Baselines3 et la plupart des bibliothèques.

- Ton siège est tiré au hasard à chaque épisode ; les autres sièges sont
  joués par ton agent lui-même (self-play), ou par `others`
  (`Agent(..., local=True, others=[baseline.act]).train(agent)`).
- Mêmes observations qu'en compétition, au format de l'agent ; une action
  invalide est remplacée par l'action neutre (`info["action_rejected"]`). Pas
  de budget de calcul à l'entraînement.
- `info["env_reward"]` : la vraie récompense ;
  `env.unwrapped.raw_observation` : l'observation d'origine (dict). De quoi
  façonner la récompense dans un wrapper Gymnasium.
- `seed` (`Agent(..., seed=0)`) : chaque environnement créé reçoit sa graine
  (seed, seed + 1...), l'entraînement est reproductible.

### Le format des observations

`observation_format = "flat"` (attribut de classe) : les observations sont un
vecteur float32 (dicts aplatis, `Discrete` en one-hot), à l'entraînement
**comme en match** — `act` reçoit toujours la même chose. Une action `Box`
y est aussi un vecteur à plat, remis à sa forme par le SDK. Par défaut
(`"dict"`), les observations gardent leur forme d'origine.

### Agents prêts à l'emploi : `frondori.agents`

Adaptés de [CleanRL](https://github.com/vwxyzjn/cleanrl) (licence MIT,
cf. `frondori/agents/THIRD_PARTY_LICENSES.md`), un fichier chacun, à lire,
copier et modifier :

| Agent | Actions | Réglages par défaut |
|---|---|---|
| `PPO` | `Discrete` ou `Box` | ceux de `ppo.py` ou `ppo_continuous_action.py` selon l'action |
| `DQN` | `Discrete` | ceux de `dqn.py` |
| `SAC` | `Box` bornées | ceux de `sac_continuous_action.py`, observations normalisées |

Tout réglage se change au constructeur (`PPO(learning_rate=1e-4,
num_envs=8)`), plus `seed`, `device`, `verbose` et `wrap_env` (un wrapper
Gymnasium appliqué à chaque environnement, par exemple pour façonner la
récompense). `learn` renvoie le retour de chaque épisode ; `save(chemin)` /
`PPO.load(chemin)` enregistrent et rechargent l'agent ; un second `train`
reprend où le premier s'est arrêté.

Exemple : en cuisine, la récompense (+1 par soupe servie) est trop rare pour
qu'un agent qui débute la découvre. Façonnée par potentiel (ce qui ne change
pas la stratégie optimale), PPO apprend à servir des soupes :

```python
import gymnasium as gym

def potential(obs):                       # progression vers une soupe servie
    held, (onions, _) = obs["self"][3], obs["pots"][0]
    return (0.15 * min(onions, 2) + 0.3 * (onions >= 2) + 0.1 * (held == 1 and onions < 2)
            + 0.2 * (held == 2 and onions >= 2) + 0.6 * (held == 3))

class KitchenShaping(gym.Wrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phi = potential(self.env.unwrapped.raw_observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        phi = potential(self.env.unwrapped.raw_observation)
        reward += 0.99 * phi - self.phi
        self.phi = phi
        return observation, reward, terminated, truncated, info

agent = PPO(wrap_env=KitchenShaping)
Agent(environment="kitchen-v0", local=True).train(agent, total_timesteps=2_000_000)
```

Sans façonnage, PPO ne sert aucune soupe ; avec, 11 soupes par match (vrai
score) après ces 2 millions de pas, environ 8 minutes sur le CPU d'un
portable. Une politique écrite à la main en sert 13.

### Ton propre algorithme

`learn` reçoit la fabrique `make_env` : à toi de créer un ou plusieurs
environnements et d'y faire tourner l'algorithme de ton choix.

```python
from frondori import Agent, Policy
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

class MyAgent(Policy):
    observation_format = "flat"

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=not self.training)
        return int(action)

    def learn(self, make_env, total_timesteps=100_000):
        self.model = SB3PPO("MlpPolicy", make_vec_env(make_env, n_envs=4))
        self.model.learn(total_timesteps)
```

Avec un script CleanRL : remplacer la création des environnements par
`envs = gym.vector.SyncVectorEnv([make_env] * num_envs)`. Les scripts CleanRL
visent Gymnasium 0.29, le SDK Gymnasium 1.x : adapter les quelques lignes des
fins d'épisode (`autoreset_mode=gym.vector.AutoresetMode.SAME_STEP`, puis
`infos["final_obs"]` au lieu de `infos["final_observation"]` ;
`infos["final_info"]` est un dict de tableaux). En self-play,
garder `SyncVectorEnv` (les adversaires sont joués par ton agent, dans le même
processus) ; `AsyncVectorEnv` copierait l'agent dans d'autres processus, qui
ne verraient pas ses progrès.

Pour un contrôle total (tous les agents, toutes les récompenses, à chaque
pas), l'environnement PettingZoo reste disponible directement :
`frondori_engine.make("kitchen-v0")`.

## Un agent écrit comme une classe

`act` peut être n'importe quel objet appelable, pas seulement une fonction :
le SDK se contente d'appeler `act(observation)` à chaque pas. Crée ton
instance une seule fois, avant le match, et passe sa méthode
(`agent.run(policy.act)`) — ou définis `__call__` et passe l'instance
elle-même (`agent.run(policy)`).

```python
from frondori import Agent

class MyPolicy:
    def __init__(self, model):
        self.model = model    # chargé une seule fois, hors du match
        self.memory = None    # conservé d'un pas à l'autre

    def act(self, observation):
        action, self.memory = self.model(observation, self.memory)
        return action

model = load_model("weights.pt")
policy = MyPolicy(model)

agent = Agent(token="frd_…", environment="kitchen-v0")
result = agent.run(policy.act)   # la même instance joue tout le match
```

- **Le temps de chargement n'est pas compté.** Seul chaque appel à `act`
  entre dans le budget de calcul : charger le modèle dans `__init__`, avant
  `run()`, ne coûte rien.
- **Échauffe ton modèle.** Le premier appel peut être bien plus lent que les
  suivants (initialisation paresseuse, compilation, allocation GPU) et
  dépasser le budget (30 ms au football) : cette action serait jouée en
  neutre. Appelle `act` une fois avant, sur une observation locale de même
  forme :

  ```python
  import frondori_engine

  env = frondori_engine.make("kitchen-v0")
  observations, _ = env.reset(seed=0)
  policy.act(observations["chef_0"])
  ```

- **Réinitialise toi-même l'état propre à un match.** `act` ne reçoit que
  l'observation, sans signal de début de match. Si ton agent garde un état
  pour la durée du match (mémoire récurrente, historique), donne à chaque
  match une nouvelle instance — le modèle, lui, reste partagé :

  ```python
  for _ in range(10):
      agent = Agent(token="frd_…", environment="kitchen-v0")
      result = agent.run(MyPolicy(model).act)
      print(result.own_return)
  ```

## Résultat d'un match

`run()`/`play()` renvoient un `MatchResult` :

- `returns` : somme des récompenses de chaque agent du match ;
  `own_return` : la tienne. Gagner, perdre, réussir ensemble... : c'est à
  toi d'interpréter selon l'environnement (compétitif ou coopératif).
- `agent_name` : l'agent que tu contrôlais (`team_0`, `chef_1`...).
- `forfeited` : agents dont le participant s'est déconnecté en cours de match.
- `actions_applied`, `actions_rejected`, `actions_too_slow`,
  `actions_missing` : le sort de tes actions, tel que rapporté par le serveur.
- `compute_budget_ms` : le temps de calcul accordé par action ;
  `compute_ms` (et `mean_compute_ms`, `max_compute_ms`) : le temps mesuré
  pour chacune de tes actions.

## Temps de calcul : ta latence réseau ne compte pas

Les matchs se jouent en pas-à-pas : le serveur attend l'action de chaque
agent avant d'avancer d'un pas. Être loin du serveur ne te coûte donc rien
(ça rallonge seulement la durée du match). Ce qui est limité, c'est le temps
de calcul de ton agent : le SDK le mesure, de la réception de l'observation
à l'envoi de ton action (décodage, `act`, encodage), et l'envoie avec elle.
Chaque environnement fixe son budget (`agent.compute_budget_ms`, connu dès le
début du match) : 30 ms au football, 200 ms en cuisine.

Le serveur confronte ce temps déclaré à ses propres mesures (temps de
réponse, aller-retour réseau) et signale les déclarations incohérentes. Tous
ces temps sont enregistrés avec le match : ils font partie des données de
recherche téléchargeables.

## Actions refusées, trop lentes ou manquantes

Une action hors de l'`action_space`, calculée en plus que le budget, ou
jamais arrivée (client bloqué, connexion coupée), ne fait pas perdre le
match : le serveur joue à la place l'action neutre de l'environnement (ne
rien faire). Mais tu en es informé : le SDK logue un avertissement à la
première occurrence de chaque cas (logger `frondori`), et le total apparaît
dans `MatchResult`.

## Erreurs

- `AuthenticationError` : token refusé, ou environnement demandé
  indisponible sur ce serveur.
- `ConnectionLostError` : connexion perdue de façon inattendue avant la fin
  du match. Une fin de match normale ne lève jamais cette erreur.
- `ProtocolError` : message qui ne respecte pas le protocole (bug serveur,
  ou version du SDK trop ancienne).

Pas de reconnexion : une déconnexion en cours de match est un forfait.

## Développer / tester le SDK

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Les tests sont isolés : aucun ne nécessite le serveur réel, ni aucun paquet
d'environnement (le mode local est testé sur l'environnement d'exemple
`tests/rps.py`, enregistré à la main). Les vecteurs de
`tests/test_messages.py` sont des octets réellement produits par le serveur
Rust (`protocol::encode`) ; s'ils cassent, le protocole a changé côté serveur.

## Publier une version

1. Mettre à jour `version` dans `pyproject.toml` et commiter.
2. Pousser un tag du même numéro : `git tag v0.1.0 && git push origin v0.1.0`.

La CI (`.github/workflows/ci.yml`) teste, construit et publie sur PyPI ; elle
refuse un tag qui ne correspond pas à la version. Publication par *Trusted
Publishing*, sans token : à configurer une fois sur PyPI (projet `frondori-sdk` >
Publishing > trusted publisher GitHub : ce dépôt, workflow `ci.yml`,
environnement `pypi`).

`frondori-engine` doit être publié AVANT ce paquet (il en dépend, et la CI
l'installe depuis PyPI).

Licence : MIT.
