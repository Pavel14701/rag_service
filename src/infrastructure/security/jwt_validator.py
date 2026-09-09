"""JWT validation using python-jose."""

from jose import jwt, JWTError

from application.interfaces import TokenValidator


class JWTValidator(TokenValidator):
    """Validator for JWT tokens."""

    def __init__(self, secret_key: str, algorithm: str = "HS256") -> None:
        self._secret = secret_key
        self._algorithm = algorithm

    def validate(self, token: str) -> dict[str, str]:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm]
            )
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
