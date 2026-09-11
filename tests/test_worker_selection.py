"""Tests for worker selection in main.py."""

import pytest

from main import select_consumers
from entrypoints.ingest_consumer import IngestConsumer
from entrypoints.query_consumer import QueryConsumer
from entrypoints.delete_consumer import DeleteConsumer
from entrypoints.reindex_consumer import ReindexConsumer

from conftest import FakeContainer


def _types(consumers):
    return [type(c) for c in consumers]


def test_select_all_runs_every_consumer():
    consumers = select_consumers("all", FakeContainer({}))
    assert set(_types(consumers)) == {
        QueryConsumer, IngestConsumer, DeleteConsumer, ReindexConsumer
    }


def test_select_query_only():
    consumers = select_consumers("query", FakeContainer({}))
    assert _types(consumers) == [QueryConsumer]


def test_select_background_excludes_query():
    consumers = select_consumers("background", FakeContainer({}))
    assert set(_types(consumers)) == {IngestConsumer, DeleteConsumer, ReindexConsumer}


def test_select_mode_is_case_insensitive():
    assert _types(select_consumers("QUERY", FakeContainer({}))) == [QueryConsumer]


def test_select_invalid_mode_raises():
    with pytest.raises(ValueError, match="WORKER_QUEUES"):
        select_consumers("nonsense", FakeContainer({}))


def test_select_empty_defaults_to_all():
    assert len(select_consumers("", FakeContainer({}))) == 4