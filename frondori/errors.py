"""Erreurs du SDK."""


class AuthenticationError(Exception):
    """Le serveur a refusé la connexion (message `AuthError` du protocole) :
    token inconnu, ou environnement demandé indisponible sur ce serveur."""


class ConnectionLostError(Exception):
    """La connexion avec le serveur a été perdue de façon INATTENDUE, avant
    d'avoir reçu un `MatchEnd` (coupure réseau, crash serveur...).

    À distinguer d'une fin de match normale : recevoir un `MatchEnd` puis
    voir le serveur fermer la connexion est le déroulement attendu et ne
    lève jamais cette erreur (cf. `Agent.play`)."""


class ProtocolError(Exception):
    """Message reçu qui ne respecte pas le protocole attendu : soit un bug
    serveur, soit une divergence de version entre le SDK et le serveur."""
