"""Tests for JWTValidator."""

import pytest
from jose import jwt

from infrastructure.security.jwt_validator import JWTValidator

SECRET = "test-secret"


def make_validator() -> JWTValidator:
    return JWTValidator(SECRET)


def test_valid_token_decoded():
    token = jwt.encode({"sub": "user-1", "role": "user"}, SECRET, algorithm="HS256")
    payload = make_validator().validate(token)
    assert payload["sub"] == "user-1"


def test_invalid_signature_raises_value_error():
    token = jwt.encode({"sub": "user-1"}, "other-secret", algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_garbage_token_raises_value_error():
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate("not-a-jwt")


def test_expired_token_raises_value_error():
    token = jwt.encode({"sub": "u", "exp": 1}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_issuer_mismatch_raises():
    validator = JWTValidator(SECRET, issuer="https://issuer.example")
    token = jwt.encode({"sub": "u", "iss": "https://other.example"}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_issuer_match_passes():
    validator = JWTValidator(SECRET, issuer="https://issuer.example")
    token = jwt.encode(
        {"sub": "u", "iss": "https://issuer.example"}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_match_passes():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode(
        {"sub": "u", "aud": "rag-service"}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_mismatch_raises():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u", "aud": "other-service"}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_missing_audience_claim_raises_when_configured():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u"}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)