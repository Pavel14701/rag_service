"""Tests for the health/metrics HTTP server."""

import json

from entrypoints.health import HealthServer
import infrastructure.observability  # noqa: F401  (registers rag_ metrics)

import pytest

pytestmark = pytest.mark.observability



async def test_healthz_returns_ok() -> None:
    server = HealthServer({}, port=0)
    status, body, content_type = await server.handle_request("GET", "/healthz")
    assert status == 200
    assert json.loads(body) == {"status": "ok"}
    assert content_type == "application/json"


async def test_readyz_ok_when_all_checks_pass():
    async def check() -> None:
        return None

    server = HealthServer({"dep": check}, port=0)
    status, body, _ = await server.handle_request("GET", "/readyz")
    assert status == 200
    assert json.loads(body)["checks"]["dep"] == "ok"


async def test_readyz_503_when_check_fails():
    async def bad_check() -> None:
        raise RuntimeError("down")

    async def good_check() -> None:
        return None

    server = HealthServer({"bad": bad_check, "good": good_check}, port=0)
    status, body, _ = await server.handle_request("GET", "/readyz")
    assert status == 503
    payload = json.loads(body)
    assert payload["status"] == "error"
    assert payload["checks"]["bad"].startswith("error: RuntimeError")
    assert payload["checks"]["good"] == "ok"


async def test_readyz_timeout_reported_as_error() -> None:
    import asyncio

    async def slow_check() -> None:
        await asyncio.sleep(10)

    server = HealthServer({"slow": slow_check}, port=0, check_timeout=0.01)
    status, body, _ = await server.handle_request("GET", "/readyz")
    assert status == 503
    assert "slow" in json.loads(body)["checks"]


async def test_metrics_endpoint_exposes_prometheus_text() -> None:
    server = HealthServer({}, port=0)
    status, body, content_type = await server.handle_request("GET", "/metrics")
    assert status == 200
    assert b"rag_" in body
    assert "text/plain" in content_type


async def test_unknown_path_returns_404() -> None:
    server = HealthServer({}, port=0)
    status, _, _ = await server.handle_request("GET", "/nope")
    assert status == 404


async def test_server_serves_http_round_trip() -> None:
    import asyncio

    server = HealthServer({}, port=0, host="127.0.0.1")
    await server.start()
    try:
        port = server._server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /healthz HTTP/1.1\r\nHost: t\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=5.0)
        writer.close()
        assert b"200" in response.split(b"\r\n")[0]
        assert b'"status": "ok"' in response
    finally:
        await server.stop()
