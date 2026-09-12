"""Tests for PII redaction of persisted conversations."""

import pytest

from domain.pii import redact_pii

pytestmark = pytest.mark.security


def test_email_redacted() -> None:
    assert redact_pii("write to john.doe@example.com please") == (
        "write to [REDACTED-EMAIL] please"
    )


def test_phone_redacted() -> None:
    result = redact_pii("call +7 (912) 345-67-89 now")
    assert "[REDACTED-PHONE]" in result
    assert "912" not in result


def test_card_number_redacted() -> None:
    result = redact_pii("card 4111 1111 1111 1111 expired")
    assert "[REDACTED-CARD]" in result
    assert "4111" not in result


def test_iban_redacted() -> None:
    result = redact_pii("transfer to DE89370400440532013000")
    assert "[REDACTED-IBAN]" in result


def test_api_key_redacted() -> None:
    result = redact_pii("my key is sk-abcdefghijklmnop1234 ok")
    assert "[REDACTED-SECRET]" in result
    assert "sk-" not in result


def test_plain_text_untouched() -> None:
    text = "The document describes quarterly revenue growth of 15 percent."
    assert redact_pii(text) == text


async def test_conversation_persisted_redacted_when_enabled(repo, vector_store, embedding, llm) -> None:
    from application.services import RetrieverService
    from domain.pii import redact_pii

    service = RetrieverService(
        vector_store, repo, embedding, llm, pii_redactor=redact_pii
    )
    await service.answer_query(
        user_id="u1", query="email john.doe@example.com about invoice"
    )
    assert len(repo.conversations) == 1
    assert "john.doe@example.com" not in repo.conversations[0]["query"]
    assert "[REDACTED-EMAIL]" in repo.conversations[0]["query"]


async def test_conversation_stored_as_is_without_redactor(repo, vector_store, embedding, llm) -> None:
    from application.services import RetrieverService

    service = RetrieverService(vector_store, repo, embedding, llm)
    await service.answer_query(user_id="u1", query="email john.doe@example.com")
    assert "john.doe@example.com" in repo.conversations[0]["query"]
