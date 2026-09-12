# Security

## Authentication (JWT)

Every queue message carries a `token` field validated before any work.

### Hardening in `JWTValidator`

- **Algorithm allowlist** — the `alg` header must equal the configured
  `JWT_ALGORITHM` before any key is touched (blocks HS256-forged tokens
  against an RS256 validator).
- **Mandatory `exp`** — tokens without expiration are rejected.
- **Issuer / audience** — verified when configured.
- **Key rotation without downtime** — `JWT_SECRET_PREVIOUS` /
  `JWT_PUBLIC_KEY_PREVIOUS` are accepted after the current key; issue with the
  new key, stop issuing with the old, drop the old after max TTL.
- **Revocation** — when a blacklist is configured every token must carry a
  `jti`; revoked ids live in `InMemoryTokenBlacklist` (single process) or
  `RedisTokenBlacklist` (shared, self-cleaning TTL = remaining token life).

### Issuing tokens

`JWTSigner` produces tokens with `sub`, `jti`, `iat`, `exp`, optional
`iss` / `aud` and a `groups` claim used by the ACL filter. Use RS256 with a
private key PEM for production.

## Authorization (ACL inside the vector store)

Every search is filtered in Qdrant:

```
should: [ owner_id == user, access_group in user_groups ]
```

Groups come from the JWT `groups` claim or the `user_groups` table
(`scripts/manage_groups.py`). Documents the user cannot see are filtered out
by the database — they never reach the prompt or the answer. The partial index
`ix_documents_owner_active` keeps owner lookups fast on large corpora.

## PII redaction

`ANONYMIZE_CONVERSATIONS=true` applies `domain/pii.py::redact_pii` to the
question and the answer before `save_conversation`: emails, phone numbers,
payment cards, IBANs and API-key-like secrets are masked with
`[REDACTED-*]` placeholders. Sources (document metadata) are not affected.

## Transport and data plane

- RabbitMQ: use `amqps://`; Qdrant/MinIO/Postgres: TLS via
  `QDRANT_HTTPS` / `MINIO_SECURE` (+ `QDRANT_API_KEY`).
- Production data only on managed devices (see `it_security.md` in the eval
  corpus for the policy text used in evals).
- The semantic answer cache is **off by default**. When enabled, it uses
  cache-then-validate: a hit is only served after every source chunk of the
  cached answer is re-checked against the requester ACL filter
  (`SEMANTIC_CACHE_STRICT_ACL=true` adds an exact access-group-set match for
  answer-level sensitivity).
