"""Interface for JWT validation."""

from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class TokenValidator(Protocol):
    """Abstract interface for validating authentication tokens."""

    def validate(self, token: str) -> Dict[str, str]:
        """Validate a JWT token and extract payload.

        Args:
            token: JWT string.

        Returns:
            Dictionary with token claims (e.g., 'sub' for user_id).

        Raises:
            ValueError: If token is invalid or expired.

        """
        ...
