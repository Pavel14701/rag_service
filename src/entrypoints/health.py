"""HTTP server exposing health checks and Prometheus metrics.

Serves, on a single port:
- ``/healthz``  — liveness probe (always OK if the server is running);
- ``/readyz``   — readiness probe with per-dependency checks;
- ``/metrics``  — Prometheus exposition (text format).
"""

import asyncio
import json
from typing import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

Check = Callable[[], Awaitable[None]]


class HealthServer:
    """Minimal dependency-free asyncio HTTP server for probes and metrics."""

    def __init__(
        self,
        checks: dict[str, Check] | None = None,
        port: int = 8000,
        check_timeout: float = 2.0,
        host: str = "0.0.0.0",
    ) -> None:
        self._checks = checks or {}
        self._port = port
        self._check_timeout = check_timeout
        self._host = host
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Bind the server socket and start serving in the background."""
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )

    async def stop(self) -> None:
        """Shut the server down."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            while True:  # drain request headers
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break
            parts = request_line.decode("latin-1").split()
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) > 1 else "/"
            status, body, content_type = await self.handle_request(method, path)
            payload = b"" if method == "HEAD" else body
            writer.write(
                (
                    f"HTTP/1.1 {status}\r\n"
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("latin-1")
                + payload
            )
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError, ValueError):
            pass  # malformed or aborted connections are ignored
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, RuntimeError):
                pass

    async def handle_request(self, method: str, path: str) -> tuple[int, bytes, str]:
        """Return (HTTP status, body, content-type) for a request.

        Public for testability — does not require a real socket.
        """
        if path == "/healthz":
            return 200, json.dumps({"status": "ok"}).encode(), "application/json"
        if path == "/readyz":
            return await self._readiness()
        if path == "/metrics":
            return 200, generate_latest(), CONTENT_TYPE_LATEST
        return 404, json.dumps({"error": "not found"}).encode(), "application/json"

    async def _readiness(self) -> tuple[int, bytes, str]:
        async def run_check(name: str, check: Check) -> tuple[str, str]:
            try:
                await asyncio.wait_for(check(), self._check_timeout)
                return name, "ok"
            except Exception as e:  # noqa: BLE001 - report any failure
                return name, f"error: {type(e).__name__}: {e}"

        results = await asyncio.gather(
            *(run_check(name, check) for name, check in self._checks.items())
        )
        checks = dict(results)
        ok = all(value == "ok" for value in checks.values())
        body = json.dumps(
            {"status": "ok" if ok else "error", "checks": checks}
        ).encode()
        return (200 if ok else 503), body, "application/json"


def build_dependency_checks(
    container, settings
) -> dict[str, Check]:
    """Build readiness checks for the external dependencies.

    Resolves concrete adapters lazily from the DI container so the
    checks are constructed only when the server probes them.
    """
    import aio_pika
    from minio import Minio
    from qdrant_client import QdrantClient
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine

    async def check_qdrant() -> None:
        client: QdrantClient = await container.get(QdrantClient)
        await asyncio.to_thread(client.get_collections)

    async def check_postgres() -> None:
        engine: AsyncEngine = await container.get(AsyncEngine)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def check_minio() -> None:
        client: Minio = await container.get(Minio)
        exists = await asyncio.to_thread(client.bucket_exists, settings.minio_bucket)
        if not exists:
            raise RuntimeError(f"bucket {settings.minio_bucket!r} does not exist")

    async def check_rabbitmq() -> None:
        connection = await asyncio.wait_for(
            aio_pika.connect(settings.rabbitmq_url), settings.health_check_timeout
        )
        await connection.close()

    return {
        "qdrant": check_qdrant,
        "postgres": check_postgres,
        "minio": check_minio,
        "rabbitmq": check_rabbitmq,
    }