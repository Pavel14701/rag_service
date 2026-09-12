from entrypoints.consumers import BaseConsumer

"""Tests for worker selection in main.py."""

import pytest

from main import select_consumers
from entrypoints.consumers import IngestConsumer
from entrypoints.consumers import QueryConsumer
from entrypoints.consumers import DeleteConsumer
from entrypoints.consumers import ReindexConsumer

from conftest import FakeContainer

pytestmark = pytest.mark.consumers


def _types(consumers: list[BaseConsumer]):
    return [type(c) for c in consumers]


def test_select_all_runs_every_consumer() -> None:
    consumers = select_consumers("all", FakeContainer({}))
    assert set(_types(consumers)) == {
        QueryConsumer, IngestConsumer, DeleteConsumer, ReindexConsumer
    }


def test_select_query_only() -> None:
    consumers = select_consumers("query", FakeContainer({}))
    assert _types(consumers) == [QueryConsumer]


def test_select_background_excludes_query() -> None:
    consumers = select_consumers("background", FakeContainer({}))
    assert set(_types(consumers)) == {IngestConsumer, DeleteConsumer, ReindexConsumer}


def test_select_mode_is_case_insensitive() -> None:
    assert _types(select_consumers("QUERY", FakeContainer({}))) == [QueryConsumer]


def test_select_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="WORKER_QUEUES"):
        select_consumers("nonsense", FakeContainer({}))


def test_select_empty_defaults_to_all() -> None:
    assert len(select_consumers("", FakeContainer({}))) == 4
