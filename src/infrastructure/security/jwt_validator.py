"""JWT validation using PyJWT.

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
- **Revocation**: when a ``blacklist`` is configured, tokens must carry
  a ``jti`` claim and are rejected if it has been revoked.
"""

from typing import Protocol, Sequence

import jwt

from application.interfaces import TokenValidator

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
    """Validator for JWT tokens.

    Optionally enforces ``iss`` and ``aud`` claims when the issuer /
    audience are configured (e.g. via settings).
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
