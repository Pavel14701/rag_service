"""GraphRAG: LLM-based relation extraction from free text."""

import json
import re

from application.interfaces import ExtractedRelation, LLMGenerator


class LLMEntityExtractor:
    """Extracts (subject; predicate; object) triples via the LLM.

    Degrades gracefully: any failure returns an empty list, so neither
    indexing nor retrieval breaks because of the graph.
    """

    _SYSTEM_PROMPT = (
        'Extract factual relations from the text as (subject; predicate; '
        'object) triples. Reply with a JSON array ONLY, e.g. '
        '[{"subject": "Atlas", "predicate": "uses", "object": '
        '"LIDAR"}]. Use an empty array when there are no clear factual '
        'relations.'
    )

    def __init__(self, llm: LLMGenerator, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    async def extract(self, text: str) -> list[ExtractedRelation]:
        """Extract triples; empty list on any failure."""
        try:
            raw = await self._llm.generate(
                self._SYSTEM_PROMPT, text, self._temperature
            )
            return self._parse_triples(raw)
        except Exception:  # noqa: BLE001 - extraction is best-effort
            return []

    @staticmethod
    def _parse_triples(raw: str) -> list[ExtractedRelation]:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return []
        if not isinstance(data, list):
            return []
        triples: list[ExtractedRelation] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subject = item.get('subject')
            predicate = item.get('predicate')
            obj = item.get('object')
            if subject and predicate and obj:
                triples.append(
                    ExtractedRelation(
                        subject=str(subject),
                        predicate=str(predicate),
                        obj=str(obj),
                    )
                )
        return triples


class NoOpEntityExtractor:
    """Pass-through extractor used when the graph is disabled."""

    async def extract(self, text: str) -> list[ExtractedRelation]:
        """No relations: disables the graph expansion path."""
        return []
