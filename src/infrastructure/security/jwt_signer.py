"""JWT issuance utility.

Companion to :mod:`jwt_validator`: signs tokens with the configured
key and algorithm (symmetric HS* or asymmetric RS*/ES* via a private
key PEM). Every issued token carries:

- ``jti`` — unique token ID, so it can be revoked via the blacklist;
- ``iat`` / ``exp`` — issued-at and expiration timestamps;
- optional ``iss`` / ``aud`` and a ``groups`` claim for access filtering.

Typical usage (e.g. in an auth service or a dev script)::

    signer = JWTSigner(private_key, algorithm="RS256", audience="rag-service")
    token = signer.issue("user-1", groups=["team-a"], expires_in=3600)
"""

import time
import uuid
from typing import Any

from jose import jwt


class JWTSigner:
    """Issues signed JWTs with revocable IDs and timestamps."""

    def __init__(
        self,
        key: str,
        algorithm: str = "HS256",
        issuer: str | None = None,
        audience: str | None = None,
        default_ttl: int = 3600,
    ) -> None:
        if algorithm not in (
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
        ):
            raise ValueError(f"Unsupported JWT algorithm: {algorithm}")
        self._key = key
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience
        self._default_ttl = default_ttl

    def issue(
        self,
        user_id: str,
        groups: list[str] | None = None,
        expires_in: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Sign a token for ``user_id``.

        Args:
            user_id: Value for the ``sub`` claim.
            groups: Optional access groups for vector-store filtering.
            expires_in: Token lifetime in seconds (default 1 hour).
            extra: Additional private claims merged into the payload.
        """
        now = int(time.time())
        ttl = expires_in if expires_in is not None else self._default_ttl
        payload: dict[str, Any] = {
            "sub": user_id,
            "jti": uuid.uuid4().hex,
            "iat": now,
            "exp": now + ttl,
        }
        if self._issuer is not None:
            payload["iss"] = self._issuer
        if self._audience is not None:
            payload["aud"] = self._audience
        if groups:
            payload["groups"] = list(groups)
        if extra:
            payload.update(extra)
        return jwt.encode(payload, self._key, algorithm=self._algorithm)