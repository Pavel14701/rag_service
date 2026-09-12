"""Tests for isolated OCR parsing: gate, subprocess runner, timeouts."""

import json
import sys
from pathlib import Path

import pytest

from application.services import IndexerService, PermanentIndexingError
from domain.model import ParseTimeoutError
from infrastructure.parsing import (
    DocumentParserSelector,
    PDFParser,
    SubprocessParseRunner,
    UnstructuredParser,
    _pdf_has_text_layer,
    _run_subprocess_json,
)

pytestmark = pytest.mark.parsing


# ---------- text-layer gate in the selector ----------


def test_text_layer_pdf_uses_fast_in_process_strategy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys.modules['infrastructure.parsing'],
        '_pdf_has_text_layer',
        lambda p: True,
    )
    selector = DocumentParserSelector(parse_isolation_enabled=True)

    parser = selector.get_parser(tmp_path / 'doc.pdf')

    assert isinstance(parser, PDFParser)
    assert parser.strategy == 'fast'
    assert parser.requires_isolation is False  # stays in-process


def test_scanned_pdf_keeps_isolated_auto_strategy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys.modules['infrastructure.parsing'],
        '_pdf_has_text_layer',
        lambda p: False,
    )
    selector = DocumentParserSelector(parse_isolation_enabled=True)

    parser = selector.get_parser(tmp_path / 'doc.pdf')

    assert parser.strategy == 'auto'
    assert parser.requires_isolation is True  # must go to subprocess


def test_probe_failure_fails_safe_to_isolation(monkeypatch, tmp_path):
    def boom(p):
        raise RuntimeError('corrupted pdf')

    monkeypatch.setattr(
        sys.modules['infrastructure.parsing'], '_pdf_has_text_layer', boom
    )
    selector = DocumentParserSelector(parse_isolation_enabled=True)

    parser = selector.get_parser(tmp_path / 'doc.pdf')

    # gate errors must not change parsing semantics: auto + isolated
    assert parser.strategy == 'auto'
    assert parser.requires_isolation is True


def test_gate_disabled_keeps_legacy_behavior(monkeypatch, tmp_path):
    def fail(p):
        raise AssertionError('gate must not run when isolation is off')

    monkeypatch.setattr(
        sys.modules['infrastructure.parsing'], '_pdf_has_text_layer', fail
    )
    selector = DocumentParserSelector(pdf_ocr_strategy='auto')

    parser = selector.get_parser(tmp_path / 'doc.pdf')

    assert parser.strategy == 'auto'  # untouched


def test_probe_absent_pypdf_returns_false(monkeypatch, tmp_path):
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def fake_import(name, *args, **kwargs):
        if name == 'pypdf':
            raise ImportError('no pypdf')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)
    assert _pdf_has_text_layer(tmp_path / 'whatever.pdf') is False


# ---------- parser isolation markers ----------


def test_parser_isolation_markers():
    assert UnstructuredParser.requires_isolation is True
    from infrastructure.parsing import DocxParser, MarkdownParser

    assert MarkdownParser.requires_isolation is False
    assert DocxParser.requires_isolation is False
    assert PDFParser().requires_isolation is True  # fail-safe default
    assert PDFParser(strategy='fast', requires_isolation=False).requires_isolation is False
    assert PDFParser(strategy='auto').requires_isolation is True


# ---------- subprocess runner (real processes) ----------


async def test_runner_returns_json_from_worker(tmp_path):
    out = tmp_path / 'out.json'
    worker = tmp_path / 'fake_worker.py'
    worker.write_text(
        'import json, sys\n'
        'from pathlib import Path\n'
        'Path(sys.argv[2]).write_text(json.dumps([{"text": "hi"}]))\n',
        encoding='utf-8',
    )

    result = await _run_subprocess_json(
        [str(worker), 'ignored', str(out)], timeout=30.0, cwd=tmp_path
    )

    assert result == [{'text': 'hi'}]
    assert not out.exists()  # temp output cleaned up


async def test_runner_raises_parse_timeout_and_kills(tmp_path):
    worker = tmp_path / 'sleeper.py'
    worker.write_text('import time; time.sleep(60)\n', encoding='utf-8')
    out = tmp_path / 'never.json'

    with pytest.raises(ParseTimeoutError):
        await _run_subprocess_json(
            [str(worker), str(out)], timeout=1.0, cwd=tmp_path
        )
    assert not out.exists()


async def test_runner_raises_on_nonzero_exit(tmp_path):
    worker = tmp_path / 'failer.py'
    worker.write_text(
        'import sys; sys.stderr.write("boom"); sys.exit(3)\n',
        encoding='utf-8',
    )

    with pytest.raises(RuntimeError, match='boom'):
        await _run_subprocess_json(
            [str(worker), str(tmp_path / 'out.json')],
            timeout=30.0,
            cwd=tmp_path,
        )


async def test_runner_targets_packaged_worker_module():
    '''The packaged worker stays import-light and validates argc.'''
    from infrastructure import parsing_worker

    # CLI contract: wrong argc exits 2 without importing unstructured
    assert parsing_worker.main(['x']) == 2
    assert 'unstructured' not in sys.modules

# ---------- IndexerService wiring ----------


class _IsolatedParser(UnstructuredParser):
    """Parser double flagged as isolation-required with canned output."""

    strategy = 'auto'
    languages = 'eng'

    def __init__(self) -> None:
        super().__init__()

    def parse(self, file_path):  # pragma: no cover - must not be called
        raise AssertionError('must be routed to the isolated runner')


class FakeRunner:
    def __init__(self, elements):
        self.elements = elements
        self.calls = []

    async def run(self, parser, file_path, timeout):
        self.calls.append((parser, file_path, timeout))
        return self.elements


class TimeoutRunner:
    async def run(self, parser, file_path, timeout):
        raise ParseTimeoutError('killed')


def _make_indexer(runner):
    return IndexerService(
        file_storage=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
        repo=None,  # type: ignore[arg-type]
        embedding=None,  # type: ignore[arg-type]
        parser_selector=None,  # type: ignore[arg-type]
        parse_timeout=5.0,
        parse_runner=runner,
    )


async def test_isolation_flag_routes_to_runner(tmp_path):
    runner = FakeRunner([{'text': 'x', 'metadata': {}}])
    indexer = _make_indexer(runner)

    elements = await indexer._parse(_IsolatedParser(), tmp_path / 'f.pdf')

    assert elements == [{'text': 'x', 'metadata': {}}]
    parser_arg, path_arg, timeout_arg = runner.calls[0]
    assert timeout_arg == 5.0
    assert path_arg == tmp_path / 'f.pdf'


async def test_timeout_in_runner_is_permanent(tmp_path):
    indexer = _make_indexer(TimeoutRunner())

    with pytest.raises(PermanentIndexingError):
        await indexer._parse(_IsolatedParser(), tmp_path / 'f.pdf')


async def test_plain_parser_stays_in_process(tmp_path, monkeypatch):
    from infrastructure.parsing import MarkdownParser

    called = []

    class FakeRunner2(FakeRunner):
        async def run(self, parser, file_path, timeout):
            called.append(1)
            return []

    indexer = _make_indexer(FakeRunner2([]))
    md = tmp_path / 'doc.md'
    md.write_text('# hi', encoding='utf-8')

    # MarkdownParser.requires_isolation is False: no runner involvement
    elements = await indexer._parse(MarkdownParser(), md)
    assert called == []
    assert elements  # parsed in-process
