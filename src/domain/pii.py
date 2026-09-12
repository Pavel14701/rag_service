"""PII redaction for persisted data (conversations).

Applies conservative, high-precision patterns so ordinary text is not
damaged, while the most common sensitive identifiers are masked before
rows are written to the ``conversations`` table:

- email addresses,
- phone numbers (international / local formats),
- payment card numbers (13–19 digits with separators),
- IBANs,
- API keys / bearer-like secrets (``sk-…``, ``Bearer …``, long hex/base64).
"""

import re

_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_PHONE = re.compile(r'(?<![\w-])\+?\d[\d\s().-]{7,}\d(?![\w-])')
_CARD = re.compile(r'(?<!\d)(?:\d[ -]?){13,19}\d(?!\d)')
_IBAN = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b')
_SECRET = re.compile(
    r'(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,}|'
    r'\b[0-9a-fA-F]{32,}\b)'
)

_PATTERNS = (
    (_SECRET, '[REDACTED-SECRET]'),
    (_IBAN, '[REDACTED-IBAN]'),
    (_CARD, '[REDACTED-CARD]'),
    (_EMAIL, '[REDACTED-EMAIL]'),
    (_PHONE, '[REDACTED-PHONE]'),
)


def redact_pii(text: str) -> str:
    """Mask well-known PII patterns in ``text`` (order matters)."""
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result
