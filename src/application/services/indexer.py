"""Document indexing service (use case)."""

import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any

from domain.entities.document import DocStatus
from domain.exceptions import DocumentNotFoundError, IndexingError
from application.interfaces import (
    FileStorage,
    VectorStore,
    DocumentRepository,
    EmbeddingModel,
)
from infrastructure.parsing.factory import ParserFactory


class IndexerService:
    """Service for indexing documents into the vector store."""

    def __init__(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
    ) -> None:
        self._file_storage = file_storage
        self._vector_store = vector_store
        self._repo = repo
        self._embedding = embedding

    async def index_document(self, doc_id: uuid.UUID) -> None:
        """
        Index a document by its ID.

        Downloads the file, parses, chunks, generates embeddings,
        and stores vectors in the vector database.

        Args:
            doc_id: UUID of the document to index.

        Raises:
            DocumentNotFoundError: If document does not exist or is deleted.
            IndexingError: If any step fails.
        """
        doc = await self._repo.get_document(doc_id)
        if doc is None or doc.deleted:
            raise DocumentNotFoundError(f"Document {doc_id} not found or deleted")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / doc.file_name
                await self._file_storage.download_file(doc.file_path, local_path)

                parser = ParserFactory.get_parser(local_path)
                elements = parser.parse(local_path)

                chunks = self._prepare_chunks(elements, doc_id, doc)
                if not chunks:
                    raise IndexingError("No chunks extracted from document")

                texts = [c["text"] for c in chunks]
                embeddings = await self._embedding.embed(texts)

                await self._vector_store.upsert(
                    ids=[c["chunk_id"] for c in chunks],
                    vectors=embeddings,
                    payloads=[c["metadata"] for c in chunks],
                )

                await self._repo.update_status(doc_id, DocStatus.INDEXED)

        except Exception as e:
            await self._repo.update_status(doc_id, DocStatus.FAILED)
            raise IndexingError(f"Indexing failed: {e}") from e

    def _prepare_chunks(
        self,
        elements: List[Dict[str, Any]],
        doc_id: uuid.UUID,
        doc: Any,  # Domain document entity
        max_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """Split parsed elements into chunks and add metadata."""
        chunks = []
        for idx, el in enumerate(elements):
            text = el.get("text", "").strip()
            if not text:
                continue
            chunk_texts = self._split_text(text, max_tokens)
            for chunk_idx, chunk_text in enumerate(chunk_texts):
                chunks.append({
                    "chunk_id": f"{doc_id}_{idx}_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": str(doc_id),
                        "owner_id": doc.owner_id,
                        "access_group": doc.access_group or "",
                        "page": el.get("metadata", {}).get("page", 0),
                        "header": el.get("metadata", {}).get("header", ""),
                    },
                })
        return chunks

    @staticmethod
    def _split_text(text: str, max_tokens: int = 512) -> List[str]:
        """Split text by words to roughly respect max_tokens."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for w in words:
            if current_len + len(w) < max_tokens:
                current.append(w)
                current_len += len(w)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
        if current:
            chunks.append(" ".join(current))
        return chunks