"""JWT validation using python-jose."""

from jose import jwt, JWTError

from application.interfaces import TokenValidator


class JWTValidator(TokenValidator):
    """Validator for JWT tokens.

    Optionally enforces ``iss`` and ``aud`` claims when the issuer /
    audience are configured (e.g. via settings).
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: str | None = None,
        audience: str | None = None,
    ) -> None:
        self._secret = secret_key
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience

    def validate(self, token: str) -> dict[str, str]:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
            )
            # python-jose skips aud validation when the claim is absent,
            # so enforce its presence explicitly when configured.
            if self._audience and "aud" not in payload:
                raise JWTError("Token is missing the audience claim")
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
