"""Isolated parsing worker: runs ``unstructured`` in a child process.

Invoked as ``python -m infrastructure.parsing_worker`` with arguments
``<file> <strategy> <languages> <out.json>`` by the
:class:`infrastructure.parsing.SubprocessParseRunner`.

Running the parser in a child process gives the parent a real OS process
to kill: a hung tesseract (OCR) never blocks a worker thread forever -
the whole process tree is terminated on timeout. The child stays dumb on
purpose: it receives strategy/languages from argv and writes the parsed
elements as JSON to ``out.json`` (never to stdout, which belongs to
third-party warnings).
"""

import json
import sys
from pathlib import Path
from typing import Any


def build_elements(
    file_path: str,
    strategy: str,
    languages: str,
) -> list[dict[str, Any]]:
    """Parse ``file_path`` into the service-wide element dicts."""
    path = Path(file_path)
    langs = [lang for lang in languages.split(',') if lang]
    if path.suffix.lower() == '.pdf':
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=str(path),
            extract_images_in_pdf=False,
            infer_table_structure=True,
            strategy=strategy,
            languages=langs or None,
        )
    else:
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(path),
            infer_table_structure=True,
            strategy='auto',
        )
    result: list[dict[str, Any]] = []
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
        el_type = str(metadata.get('category') or 'text').lower()
        el_metadata: dict[str, Any] = {
            'page': metadata.get('page_number', 0),
            'header': metadata.get('header', ''),
            'footer': metadata.get('footer', ''),
            'type': el_type,
        }
        table_html = metadata.get('text_as_html')
        if el_type == 'table' and table_html:
            el_metadata['table_html'] = table_html
        result.append({'text': text, 'metadata': el_metadata})
    return result


def main(argv: list[str]) -> int:
    """CLI entry: parse and write JSON to the output path."""
    if len(argv) != 5:
        print(
            'usage: parsing_worker <file> <strategy> <languages> <out>',
            file=sys.stderr,
        )
        return 2
    file_path, strategy, languages, out_path = argv[1:]
    elements = build_elements(file_path, strategy, languages)
    Path(out_path).write_text(
        json.dumps(elements, ensure_ascii=False), encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
