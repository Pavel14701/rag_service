"""Tests for JWTSigner (issue side of the JWT infrastructure)."""

import pytest
import jwt as jose_jwt

from infrastructure.security import JWTSigner
from infrastructure.security import JWTValidator

from test_jwt_validator import SECRET, PRIVATE_KEY_PEM, PUBLIC_KEY_PEM

pytestmark = pytest.mark.security


def test_hs256_roundtrip_contains_security_claims() -> None:
    signer = JWTSigner(SECRET)
    token = signer.issue("user-1", groups=["team-a"])
    payload = JWTValidator(SECRET).validate(token)

    assert payload["sub"] == "user-1"
    assert payload["groups"] == ["team-a"]
    # every issued token is revocable and time-bounded
    assert payload["jti"]
    assert payload["iat"] <= payload["exp"]
    assert payload["exp"] - payload["iat"] == 3600  # default TTL


def test_rs256_sign_and_validate_roundtrip() -> None:
    signer = JWTSigner(PRIVATE_KEY_PEM, algorithm="RS256", audience="rag-service")
    validator = JWTValidator(PUBLIC_KEY_PEM, algorithm="RS256", audience="rag-service")

    token = signer.issue("user-2", expires_in=600)
    payload = validator.validate(token)

    assert payload["sub"] == "user-2"
    assert payload["aud"] == "rag-service"
    assert payload["exp"] - payload["iat"] == 600


def test_rs256_token_forged_with_hs256_rejected() -> None:
    signer = JWTSigner(PRIVATE_KEY_PEM, algorithm="RS256")
    token = signer.issue("u")
    validator = JWTValidator(PUBLIC_KEY_PEM, algorithm="RS256")
    assert validator.validate(token)["sub"] == "u"

    # HS256-signed token (alg-confusion attempt) is rejected by the
    # algorithm allowlist before any key material is used.
    import base64, json

    header, body, _ = token.split(".")
    forged_payload = json.loads(
        base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    )
    forged = jose_jwt.encode(forged_payload, "attacker-hmac-secret-0123456789abcdef", algorithm="HS256")
    with pytest.raises(ValueError, match="unexpected algorithm"):
        validator.validate(forged)


def test_issuer_and_audience_embedded_when_configured() -> None:
    signer = JWTSigner(SECRET, issuer="https://auth.example", audience="rag-service")
    payload = JWTValidator(
        SECRET, issuer="https://auth.example", audience="rag-service"
    ).validate(signer.issue("u"))
    assert payload["iss"] == "https://auth.example"
    assert payload["aud"] == "rag-service"


def test_jti_is_unique_per_token() -> None:
    signer = JWTSigner(SECRET)
    jti1 = JWTValidator(SECRET).validate(signer.issue("u"))["jti"]
    jti2 = JWTValidator(SECRET).validate(signer.issue("u"))["jti"]
    assert jti1 != jti2


def test_extra_claims_merged() -> None:
    signer = JWTSigner(SECRET)
    payload = JWTValidator(SECRET).validate(
        signer.issue("u", extra={"role": "admin"})
    )
    assert payload["role"] == "admin"


def test_unsupported_algorithm_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported JWT algorithm"):
        JWTSigner(SECRET, algorithm="none")
