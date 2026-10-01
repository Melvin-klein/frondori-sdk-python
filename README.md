# frondori-sdk

SDK Python pour connecter un modèle à la plateforme de compétition Frondori.
Il gère la connexion WebSocket, l'authentification et le protocole réseau :
il ne reste qu'à écrire une fonction `observation -> action`.

Le SDK ne connaît aucun jeu en particulier : tu choisis l'environnement à
jouer (football, cuisine coopérative...) à chaque connexion, et le serveur
décrit ses observations et ses actions au début de chaque match. Un même
agent (un même token) peut jouer à plusieurs environnements ; son classement
est tenu séparément pour chacun.

Pour s'entraîner en local, sans serveur, utiliser `frondori-engine` : ce sont
les mêmes environnements, et **les observations reçues en compétition ont
exactement la même forme qu'en local**. Une politique entraînée avec
`frondori-engine` se branche donc telle quelle ici.

## Installation

```bash
pip install frondori-sdk
```

Dépendances : Python >= 3.10, `websockets`, `msgpack`, `numpy`, `gymnasium`.

## Démarrage rapide

```python
from frondori import Agent

agent = Agent(url="wss://…/agent", token="frd_…", environment="kitchen-v0")

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

agent = Agent(url="ws://localhost:8080/agent", token="frd_…", environment="kitchen-v0")
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
      agent = Agent(url="ws://localhost:8080/agent", token="frd_…", environment="kitchen-v0")
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

Les tests sont isolés : aucun ne nécessite le serveur réel. Les vecteurs de
`tests/test_messages.py` sont des octets réellement produits par le serveur
Rust (`protocol::encode`) ; s'ils cassent, le protocole a changé côté serveur.
