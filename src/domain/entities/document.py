"""Document entity and related value objects."""

import uuid
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class DocStatus(str, Enum):
    """Possible statuses of a document."""

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class Document:
    """Domain entity representing a document."""

    id: uuid.UUID
    file_name: str
    file_path: str
    file_hash: str
    owner_id: str
    access_group: str | None
    uploaded_at: datetime
    status: DocStatus
    deleted: bool = False
