"""Tests for JWTValidator."""

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt

from infrastructure.security import JWTValidator

pytestmark = pytest.mark.security

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
LONG = "x" * 40

_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_KEY_PEM = _key.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
PUBLIC_KEY_PEM = _key.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()


def make_validator() -> JWTValidator:
    return JWTValidator(SECRET)


def test_valid_token_decoded() -> None:
    token = jwt.encode({"sub": "user-1", "role": "user", "exp": 9999999999}, SECRET, algorithm="HS256")
    payload = make_validator().validate(token)
    assert payload["sub"] == "user-1"


def test_invalid_signature_raises_value_error() -> None:
    token = jwt.encode({"sub": "user-1"}, f"other-secret-{LONG}", algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_garbage_token_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate("not-a-jwt")


def test_expired_token_raises_value_error() -> None:
    token = jwt.encode({"sub": "u", "exp": 1}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_issuer_mismatch_raises() -> None:
    validator = JWTValidator(SECRET, issuer="https://issuer.example")
    token = jwt.encode({"sub": "u", "iss": "https://other.example", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_issuer_match_passes() -> None:
    validator = JWTValidator(SECRET, issuer="https://issuer.example")
    token = jwt.encode(
        {"sub": "u", "iss": "https://issuer.example", "exp": 9999999999}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_match_passes() -> None:
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode(
        {"sub": "u", "aud": "rag-service", "exp": 9999999999}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_mismatch_raises() -> None:
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u", "aud": "other-service", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_missing_audience_claim_raises_when_configured() -> None:
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="audience"):
        validator.validate(token)


# ---------- hardening ----------


def test_token_without_exp_rejected() -> None:
    token = jwt.encode({"sub": "u"}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_expired_token_rejected_even_with_other_valid_claims() -> None:
    import time

    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"
    )
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_algorithm_confusion_rejected() -> None:
    # Token signed with HS256 against an RS256-configured validator
    # (attacker trick: sign with the public key as an HMAC secret).
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    validator = JWTValidator(PUBLIC_KEY_PEM, algorithm="RS256")
    with pytest.raises(ValueError, match="unexpected algorithm"):
        validator.validate(token)


def test_key_rotation_accepts_previous_secret() -> None:
    new_secret, old_secret = f"new-secret-{LONG}", f"old-secret-{LONG}"
    validator = JWTValidator([new_secret, old_secret])
    old_token = jwt.encode(
        {"sub": "u", "exp": 9999999999}, old_secret, algorithm="HS256"
    )
    new_token = jwt.encode(
        {"sub": "u", "exp": 9999999999}, new_secret, algorithm="HS256"
    )
    assert validator.validate(old_token)["sub"] == "u"
    assert validator.validate(new_token)["sub"] == "u"


def test_unknown_secret_rejected_during_rotation() -> None:
    validator = JWTValidator([f"new-secret-{LONG}", f"old-secret-{LONG}"])
    token = jwt.encode({"sub": "u", "exp": 9999999999}, "attacker-" + "x" * 32, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_revoked_jti_rejected() -> None:
    from infrastructure.security import InMemoryTokenBlacklist

    blacklist = InMemoryTokenBlacklist()
    validator = JWTValidator(SECRET, blacklist=blacklist)
    token = jwt.encode(
        {"sub": "u", "exp": 9999999999, "jti": "tok-1"}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"

    blacklist.revoke("tok-1", ttl=3600)
    with pytest.raises(ValueError, match="revoked"):
        validator.validate(token)


def test_blacklist_configured_requires_jti() -> None:
    from infrastructure.security import InMemoryTokenBlacklist

    validator = JWTValidator(SECRET, blacklist=InMemoryTokenBlacklist())
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="missing jti"):
        validator.validate(token)


def test_unsupported_algorithm_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unsupported JWT algorithm"):
        JWTValidator(SECRET, algorithm="HS1024")


def test_rotated_token_blacklist_shared_across_keys() -> None:
    from infrastructure.security import InMemoryTokenBlacklist

    blacklist = InMemoryTokenBlacklist()
    validator = JWTValidator(
        [f"new-{LONG}", f"old-{LONG}"], blacklist=blacklist
    )
    token = jwt.encode(
        {"sub": "u", "exp": 9999999999, "jti": "j-1"}, f"old-{LONG}", algorithm="HS256"
    )
    blacklist.revoke("j-1", ttl=3600)
    with pytest.raises(ValueError, match="revoked"):
        validator.validate(token)
