"""Document parsing: format-specific parsers and parser selection.

Heavy ``unstructured`` dependencies are imported lazily *inside* the
parsers that need them: importing ``unstructured`` pulls in
python-magic/libmagic, which may be unavailable (e.g. on Windows
without libmagic installed). Lightweight dependencies (``markdown``,
``beautifulsoup4``) are imported at module level.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup

from application.interfaces import DocumentParser
from domain.model import ParseTimeoutError
from infrastructure.observability import PARSE_TIMEOUTS_TOTAL


class MarkdownParser(DocumentParser):
    """Parser for .md files that splits by headers."""

    requires_isolation = False

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse markdown, splitting on headers (h1-h3).

        Returns:
            list of chunks, each with 'text' and metadata including 'header'.

        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')

        chunks: list[dict[str, Any]] = []
        current_section: list[str] = []
        header = 'Root'

        for el in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'pre']):
            if el.name in ('h1', 'h2', 'h3'):
                if current_section:
                    chunks.append(
                        {
                            'text': '\n'.join(current_section),
                            'metadata': {'header': header},
                        }
                    )
                    current_section = []
                header = el.get_text(strip=True)
                current_section.append(header)
            else:
                current_section.append(el.get_text(strip=True))

        if current_section:
            chunks.append(
                {
                    'text': '\n'.join(current_section),
                    'metadata': {'header': header},
                }
            )

        return chunks


class PDFParser(DocumentParser):
    """Parser for PDF documents that extracts text and metadata.

    ``strategy`` controls the OCR handling for scanned documents
    (unstructured partition strategy):
    - ``auto``        — OCR when needed (default);
    - ``ocr_only``    — force OCR for all pages;
    - ``hi_res``      — high-resolution layout+OCR pipeline;
    - ``fast``        — plain text extraction, no OCR.

    ``languages`` is a comma-separated list of tesseract languages
    (e.g. ``eng`` or ``eng,rus``) used when OCR runs.
    """

    def __init__(
        self,
        strategy: str = 'auto',
        languages: str = 'eng',
        requires_isolation: bool = True,
    ) -> None:
        self._strategy = strategy
        self._languages = languages
        # Public on purpose: read by the isolated parse runner.
        self.strategy = strategy
        self.languages = languages
        # Fast text-layer parses stay in-process; OCR-capable strategies
        # must run in a killable child process (hung tesseract).
        self.requires_isolation = requires_isolation

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a PDF file and return a list of text elements with metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            list of dicts with keys 'text'
            and 'metadata' (includes page numbers).

        """
        from unstructured.partition.pdf import partition_pdf

        languages = [lang for lang in self._languages.split(',') if lang]
        elements = partition_pdf(
            filename=str(file_path),
            extract_images_in_pdf=False,
            infer_table_structure=True,
            strategy=self._strategy,
            languages=languages or None,
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
            # Multimodality: tag the element category ("table", "image",
            # "title", ...) and keep the HTML representation of tables so
            # they can be embedded and stored with structure preserved.
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
            result.append(
                {
                    'text': text,
                    'metadata': el_metadata,
                }
            )
        return result


class DocxParser(DocumentParser):
    """Parser for DOCX documents that extracts text and metadata."""

    requires_isolation = False

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a DOCX file and return a list of text elements with metadata.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            list of dicts with keys 'text' and 'metadata'
            (includes page numbers if available).

        """
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(
            filename=str(file_path),
            infer_table_structure=True,
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
            result.append(
                {
                    'text': text,
                    'metadata': {
                        'page': metadata.get('page_number', 0),
                        'category': metadata.get('category', ''),
                    },
                }
            )
        return result


class UnstructuredParser(DocumentParser):
    """Universal parser that uses unstructured's auto-detection.

    Works for many formats: .txt, .html, .epub, etc.
    """

    # auto-partition can OCR arbitrary formats: run it isolated
    requires_isolation = True

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse any supported file format and return text elements.

        Args:
            file_path: Path to the file.

        Returns:
            list of dicts with 'text' and 'metadata'.

        """
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(file_path),
            infer_table_structure=True,
            strategy='auto',
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
            result.append(
                {
                    'text': text,
                    'metadata': {
                        'page': metadata.get('page_number', 0),
                        'category': metadata.get('category', ''),
                        'filetype': metadata.get('filetype', ''),
                    },
                }
            )
        return result


# libmagic (python-magic) loads a native DLL that is broken on some
# systems: `import magic` can crash the whole interpreter with an
# access violation. Probe availability in an isolated subprocess once
# and fail open when the binding is unusable.
@cache
def _magic_available() -> bool:
    """Check (in a subprocess) that the magic binding can be imported."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import magic'],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _load_magic() -> Any:
    """Return the magic module (or a test double); None if unusable."""
    mod = sys.modules.get('magic')
    if mod is not None:
        return mod  # test doubles are injected here
    if not _magic_available():
        return None
    try:
        import magic

        return magic
    except Exception:
        return None


# Known extensions -> acceptable MIME types (as reported by libmagic).
_MIME_BY_EXT = {
    '.pdf': ('application/pdf',),
    '.docx': (
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/zip',
    ),
    '.doc': ('application/msword',),
    '.md': ('text/plain', 'text/markdown', 'text/x-markdown'),
    '.txt': ('text/plain',),
    '.html': ('text/html',),
}


class FileMimeTypeError(ValueError):
    """Real file content (magic bytes) does not match its extension."""


class DocumentParserSelector:
    """Selects a parser by file extension (implements ``ParserSelector``).

    Constructed with the OCR settings at DI-wiring time, replacing the
    former process-wide ``ParserFactory.configure()`` class state.

    For known extensions the real MIME type (magic bytes) is verified
    before a parser is returned: an ``.exe`` renamed to ``.pdf`` must
    be rejected instead of being fed to tesseract (OCR CPU hang).
    """

    def __init__(
        self,
        pdf_ocr_strategy: str = 'auto',
        pdf_ocr_languages: str = 'eng',
        parse_isolation_enabled: bool = False,
    ) -> None:
        self._pdf_ocr_strategy = pdf_ocr_strategy
        self._pdf_ocr_languages = pdf_ocr_languages
        # Text-layer gate + isolated OCR parsing (opt-in).
        self._parse_isolation_enabled = parse_isolation_enabled

    def _validate_mime(self, file_path: Path) -> None:
        """Reject files whose magic bytes contradict the extension."""
        expected = _MIME_BY_EXT.get(file_path.suffix.lower())
        if not expected or not file_path.exists():
            return  # unknown extension or in-memory path: nothing to check
        magic = _load_magic()
        if magic is None:
            return  # libmagic unavailable: fail open
        detected = magic.from_file(str(file_path), mime=True)
        if detected in expected:
            return
        if detected.startswith('text/') and any(
            e.startswith('text/') for e in expected
        ):
            return  # any textual type is fine for text-like formats
        raise FileMimeTypeError(
            f'{file_path.name}: content type {detected!r} does not match '
            f'extension {file_path.suffix!r} (expected {expected[0]!r})'
        )

    def get_parser(self, file_path: Path) -> DocumentParser:
        """Return a parser instance based on the file extension.

        Args:
            file_path: Path to the file to be parsed.

        Returns:
            A DocumentParser implementation.

        """
        self._validate_mime(file_path)
        ext = file_path.suffix.lower()
        if ext == '.md':
            return MarkdownParser()
        if ext == '.pdf':
            strategy = self._pdf_ocr_strategy
            requires_isolation = True
            has_text_layer = False
            if self._parse_isolation_enabled and strategy == 'auto':
                try:
                    has_text_layer = _pdf_has_text_layer(file_path)
                except Exception:  # noqa: BLE001 - gate must never raise
                    has_text_layer = False  # fail safe to the OCR path
            if (
                self._parse_isolation_enabled
                and strategy == 'auto'
                and has_text_layer
            ):
                # Text layer present: skip OCR entirely, keep the fast
                # pure-text parse in-process. Any probe failure above
                # fails safe to the isolated OCR-capable path.
                strategy = 'fast'
                requires_isolation = False
            return PDFParser(
                strategy=strategy,
                languages=self._pdf_ocr_languages,
                requires_isolation=requires_isolation,
            )
        return DocxParser() if ext == '.docx' else UnstructuredParser()


def _pdf_has_text_layer(
    file_path: Path,
    min_chars: int = 32,
    max_pages: int = 3,
) -> bool:
    """Cheap probe: do the first pages already carry a text layer?

    Best-effort by design: any import/parse failure returns False,
    which fails safe to the isolated OCR-capable parsing path.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return False
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages[:max_pages]:
            text = (page.extract_text() or '').strip()
            if len(text) >= min_chars:
                return True
    except Exception:  # noqa: BLE001 - probe must never break selection
        return False
    return False


async def _run_subprocess_json(
    args: list[str],
    timeout: float,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a child process that writes a JSON element list; kill on timeout.

    Kills the whole process tree (tesseract children included): on
    POSIX via a new process group + SIGKILL, on Windows via
    taskkill /F /T. Raises ParseTimeoutError past ``timeout`` and
    RuntimeError when the child exits non-zero.
    """
    kwargs: dict[str, Any] = {}
    if os.name != 'nt':
        kwargs['start_new_session'] = True
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        **kwargs,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        _kill_process_tree(proc)
        await proc.wait()
        PARSE_TIMEOUTS_TOTAL.labels(mode='isolated').inc()
        raise ParseTimeoutError(
            f'isolated parsing timed out after {timeout}s; process tree killed'
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(
            f'isolated parsing failed (exit {proc.returncode}): '
            f'{stderr.decode(errors="replace").strip()}'
        )
    out_path = Path(args[-1])
    try:
        return list(json.loads(out_path.read_text(encoding='utf-8')))
    finally:
        out_path.unlink(missing_ok=True)


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the child and all of its descendants, cross-platform."""
    if os.name != 'nt':
        # POSIX-only symbols resolved dynamically: they do not exist on
        # Windows builds and this branch never runs there.
        killpg = getattr(os, 'killpg', None)
        getpgid = getattr(os, 'getpgid', None)
        sigkill = getattr(signal, 'SIGKILL', None)
        if killpg and getpgid and sigkill:
            try:
                killpg(getpgid(proc.pid), sigkill)
            except (ProcessLookupError, PermissionError):
                pass
        return
    subprocess.run(
        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
        capture_output=True,
        check=False,
    )
    try:
        proc.kill()
    except ProcessLookupError:
        pass


class SubprocessParseRunner:
    """IsolatedParseRunner on python -m infrastructure.parsing_worker.

    The child imports the heavy unstructured stack in isolation, so a
    hung OCR costs one killed process instead of a poisoned worker
    thread pool.
    """

    async def run(
        self,
        parser: DocumentParser,
        file_path: Path,
        timeout: float,
    ) -> list[dict[str, Any]]:
        """Parse in a child process; kill the tree past ``timeout``."""
        strategy = str(getattr(parser, 'strategy', 'auto'))
        languages = str(getattr(parser, 'languages', 'eng'))
        out_path = Path(f'{file_path}.elements.json')
        args = [
            '-m',
            'infrastructure.parsing_worker',
            str(file_path),
            strategy,
            languages,
            str(out_path),
        ]
        return await _run_subprocess_json(args, timeout, cwd=_package_root())


@cache
def _package_root() -> Path:
    """Directory put on the child path so -m resolves the worker module."""
    return Path(__file__).resolve().parent.parent
