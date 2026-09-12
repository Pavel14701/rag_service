"""Security infrastructure: JWT signing, validation and revocation.

- ``JWTSigner`` issues tokens (``jti``/``iat``/``exp``, optional
  ``iss``/``aud``/``groups`` claims);
- ``JWTValidator`` verifies them with algorithm allowlisting, key
  rotation support and blacklist checks;
- ``InMemoryTokenBlacklist`` / ``RedisTokenBlacklist`` store revoked
  ``jti`` values in-process or shared via Redis.
"""

import time
from collections.abc import Sequence
import uuid
from typing import Any, Protocol

import jwt

from application.interfaces import TokenValidator


class JWTSigner:
    """Issues signed JWTs with revocable IDs and timestamps."""

    def __init__(
        self,
        key: str,
        algorithm: str = 'HS256',
        issuer: str | None = None,
        audience: str | None = None,
        default_ttl: int = 3600,
    ) -> None:
        if algorithm not in (
            'HS256',
            'HS384',
            'HS512',
            'RS256',
            'RS384',
            'RS512',
            'ES256',
        ):
            raise ValueError(f'Unsupported JWT algorithm: {algorithm}')
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
            'sub': user_id,
            'jti': uuid.uuid4().hex,
            'iat': now,
            'exp': now + ttl,
        }
        if self._issuer is not None:
            payload['iss'] = self._issuer
        if self._audience is not None:
            payload['aud'] = self._audience
        if groups:
            payload['groups'] = list(groups)
        if extra:
            payload |= extra
        token: str = jwt.encode(payload, self._key, algorithm=self._algorithm)
        return token


# Algorithms this module is designed to accept (validated in Settings too).
SUPPORTED_ALGORITHMS = (
    'HS256',
    'HS384',
    'HS512',
    'RS256',
    'RS384',
    'RS512',
    'ES256',
)


class TokenBlacklist(Protocol):
    """Revocation store for token IDs (``jti`` claims)."""

    def contains(self, jti: str) -> bool:
        """True when the given token ID has been revoked."""
        ...


class JWTValidator(TokenValidator):
    """Validator for JWT tokens (implements the application port).

    Hardening:
    - **Algorithm allowlist**: the token ``alg`` header must equal the
      configured algorithm before any signature check, preventing
      algorithm-confusion attacks (e.g. HS256-forged tokens against an
      RS256 validator).
    - **Asymmetric algorithms**: RS256 / ES256 are supported; pass the
      *public* key PEM as the verification key.
    - **Key rotation**: several keys may be configured (current first,
      previous second); a token signed with any of them is accepted, so
      secrets can be rotated without downtime.
    - **Mandatory ``exp``**: tokens without an expiration are rejected.
    - **Revocation**: when a ``blacklist`` is configured, tokens must
      carry a ``jti`` claim and are rejected if it has been revoked.
    """

    def __init__(
        self,
        keys: str | Sequence[str],
        algorithm: str = 'HS256',
        issuer: str | None = None,
        audience: str | None = None,
        blacklist: TokenBlacklist | None = None,
    ) -> None:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f'Unsupported JWT algorithm: {algorithm}')
        self._keys = [keys] if isinstance(keys, str) else list(keys)
        if not self._keys:
            raise ValueError('At least one verification key is required')
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience
        self._blacklist = blacklist

    def validate(self, token: str) -> dict[str, str]:
        """Validate a JWT and return its claims."""
        # Defense in depth: reject tokens whose alg header does not match
        # the configured algorithm before any key is touched.
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as e:
            raise ValueError(f'Invalid token: {e}')
        if header.get('alg') != self._algorithm:
            raise ValueError(
                f'Invalid token: unexpected algorithm {header.get("alg")!r}'
            )

        payload = None
        last_error: Exception | None = None
        # Try the current key first, then rotation predecessors.
        for key in self._keys:
            try:
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=[self._algorithm],
                    issuer=self._issuer,
                    # Audience is checked below (presence + equality) so
                    # behaviour stays identical regardless of whether an
                    # audience is configured.
                    options={'require': ['exp'], 'verify_aud': False},
                )
                break
            except jwt.PyJWTError as e:
                last_error = e
        if payload is None:
            raise ValueError(f'Invalid token: {last_error}')

        # PyJWT verifies ``aud`` only when the claim is present, so
        # enforce its presence and equality explicitly when configured
        # (keeps the error surface uniform for callers).
        if self._audience is not None:
            aud = payload.get('aud')
            if aud is None:
                raise ValueError(
                    'Invalid token: Token is missing the audience claim'
                )
            auds = aud if isinstance(aud, list) else [aud]
            if self._audience not in auds:
                raise ValueError(f'Invalid token: unexpected audience {aud!r}')

        if self._blacklist is not None:
            jti = payload.get('jti')
            if not jti:
                raise ValueError('Invalid token: missing jti claim')
            if self._blacklist.contains(jti):
                raise ValueError('Invalid token: token has been revoked')

        return dict(payload)


class InMemoryTokenBlacklist:
    """In-process revocation store with per-entry TTL.

    Fits single-process deployments and tests; use
    :class:`RedisTokenBlacklist` when several worker replicas share the
    revocation state.
    """

    def __init__(self) -> None:
        self._revoked: dict[str, float] = {}  # jti -> revoked-until timestamp

    def revoke(self, jti: str, ttl: float) -> None:
        """Revoke ``jti`` for ``ttl`` seconds (until the token expires)."""
        self._revoked[jti] = time.time() + ttl
        # Opportunistic cleanup of expired entries.
        now = time.time()
        expired = [k for k, until in self._revoked.items() if until <= now]
        for key in expired:
            del self._revoked[key]

    def contains(self, jti: str) -> bool:
        """True when the given token ID is revoked."""
        until = self._revoked.get(jti)
        return until is not None and until > time.time()


class RedisTokenBlacklist:
    """Redis-backed revocation store shared by all worker replicas.

    Each revoked ``jti`` is stored with an expiry equal to the token's
    remaining lifetime, so the blacklist self-cleans.
    """

    def __init__(
        self, redis_client: Any, prefix: str = 'rag:revoked-jti'
    ) -> None:
        self._client = redis_client
        self._prefix = prefix

    def revoke(self, jti: str, ttl: float) -> None:
        """Store the revoked jti in Redis until the token would expire."""
        self._client.setex(f'{self._prefix}:{jti}', max(int(ttl), 1), '1')

    def contains(self, jti: str) -> bool:
        """True when the token ID is present in Redis."""
        return bool(self._client.exists(f'{self._prefix}:{jti}'))
