"""Conversation history entity."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Conversation:
    """User conversation record."""

    id: uuid.UUID
    user_id: str
    query: str
    response: str
    sources: list[dict[str, Any]]
    created_at: datetime
