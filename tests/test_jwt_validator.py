"""Tests for JWTValidator."""

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from jose import jwt

from infrastructure.security.jwt_validator import JWTValidator

SECRET = "test-secret"

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


def test_valid_token_decoded():
    token = jwt.encode({"sub": "user-1", "role": "user", "exp": 9999999999}, SECRET, algorithm="HS256")
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
    token = jwt.encode({"sub": "u", "iss": "https://other.example", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_issuer_match_passes():
    validator = JWTValidator(SECRET, issuer="https://issuer.example")
    token = jwt.encode(
        {"sub": "u", "iss": "https://issuer.example", "exp": 9999999999}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_match_passes():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode(
        {"sub": "u", "aud": "rag-service", "exp": 9999999999}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"


def test_audience_mismatch_raises():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u", "aud": "other-service", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_missing_audience_claim_raises_when_configured():
    validator = JWTValidator(SECRET, audience="rag-service")
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="audience"):
        validator.validate(token)


# ---------- hardening ----------


def test_token_without_exp_rejected():
    token = jwt.encode({"sub": "u"}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_expired_token_rejected_even_with_other_valid_claims():
    import time

    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"
    )
    with pytest.raises(ValueError, match="Invalid token"):
        make_validator().validate(token)


def test_algorithm_confusion_rejected():
    # Token signed with HS256 against an RS256-configured validator
    # (attacker trick: sign with the public key as an HMAC secret).
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    validator = JWTValidator(PUBLIC_KEY_PEM, algorithm="RS256")
    with pytest.raises(ValueError, match="unexpected algorithm"):
        validator.validate(token)


def test_key_rotation_accepts_previous_secret():
    new_secret, old_secret = "new-secret", "old-secret"
    validator = JWTValidator([new_secret, old_secret])
    old_token = jwt.encode(
        {"sub": "u", "exp": 9999999999}, old_secret, algorithm="HS256"
    )
    new_token = jwt.encode(
        {"sub": "u", "exp": 9999999999}, new_secret, algorithm="HS256"
    )
    assert validator.validate(old_token)["sub"] == "u"
    assert validator.validate(new_token)["sub"] == "u"


def test_unknown_secret_rejected_during_rotation():
    validator = JWTValidator(["new-secret", "old-secret"])
    token = jwt.encode({"sub": "u", "exp": 9999999999}, "attacker", algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid token"):
        validator.validate(token)


def test_revoked_jti_rejected():
    from infrastructure.security.token_blacklist import InMemoryTokenBlacklist

    blacklist = InMemoryTokenBlacklist()
    validator = JWTValidator(SECRET, blacklist=blacklist)
    token = jwt.encode(
        {"sub": "u", "exp": 9999999999, "jti": "tok-1"}, SECRET, algorithm="HS256"
    )
    assert validator.validate(token)["sub"] == "u"

    blacklist.revoke("tok-1", ttl=3600)
    with pytest.raises(ValueError, match="revoked"):
        validator.validate(token)


def test_blacklist_configured_requires_jti():
    from infrastructure.security.token_blacklist import InMemoryTokenBlacklist

    validator = JWTValidator(SECRET, blacklist=InMemoryTokenBlacklist())
    token = jwt.encode({"sub": "u", "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(ValueError, match="missing jti"):
        validator.validate(token)


def test_unsupported_algorithm_rejected_at_construction():
    with pytest.raises(ValueError, match="Unsupported JWT algorithm"):
        JWTValidator(SECRET, algorithm="HS1024")


def test_rotated_token_blacklist_shared_across_keys():
    from infrastructure.security.token_blacklist import InMemoryTokenBlacklist

    blacklist = InMemoryTokenBlacklist()
    validator = JWTValidator(["new", "old"], blacklist=blacklist)
    token = jwt.encode(
        {"sub": "u", "exp": 9999999999, "jti": "j-1"}, "old", algorithm="HS256"
    )
    blacklist.revoke("j-1", ttl=3600)
    with pytest.raises(ValueError, match="revoked"):
        validator.validate(token)