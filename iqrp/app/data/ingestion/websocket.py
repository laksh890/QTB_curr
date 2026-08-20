"""WebSocket ingestion engine with reconnect, heartbeat, and integrity checks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from iqrp.app.core.exceptions import DataError

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class WebsocketStats:
    messages: int = 0
    duplicates: int = 0
    sequence_gaps: int = 0
    reconnects: int = 0
    last_latency_ms: float | None = None
    packet_loss_events: int = 0


@dataclass
class WebsocketEngine:
    """Production-oriented websocket client wrapper.

    Transport is injected (``connect``) so unit tests can mock sockets without
    a live network. Production wiring uses the ``websockets`` package.
    """

    url: str
    handler: MessageHandler
    heartbeat_interval: float = 15.0
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    expected_sequence_key: str = "sequence"
    message_id_key: str = "id"
    connect: Callable[[str], Any] | None = None
    send_heartbeat: Callable[[Any], Awaitable[None]] | None = None
    stats: WebsocketStats = field(default_factory=WebsocketStats)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _seen_ids: set[str] = field(default_factory=set)
    _last_sequence: int | None = None

    async def stop(self) -> None:
        self._stop.set()

    async def run(self, *, max_reconnects: int | None = None) -> None:
        """Run until stopped or reconnect budget exhausted."""
        if self.connect is None:
            raise DataError("Websocket connect callable is not configured", code="WS_NO_CONNECT")
        delay = self.reconnect_base_delay
        reconnects = 0
        while not self._stop.is_set():
            try:
                await self._session()
                delay = self.reconnect_base_delay
            except Exception as exc:
                reconnects += 1
                self.stats.reconnects += 1
                logger.warning(
                    "ws_reconnect url={} attempt={} error={}",
                    self.url,
                    reconnects,
                    exc,
                )
                if max_reconnects is not None and reconnects > max_reconnects:
                    raise DataError(
                        f"Websocket reconnect budget exhausted: {exc}",
                        code="WS_RECONNECT_EXHAUSTED",
                    ) from exc
                await asyncio.sleep(delay)
                delay = min(self.reconnect_max_delay, delay * 2.0)

    async def _session(self) -> None:
        assert self.connect is not None
        async with self.connect(self.url) as ws:
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    message = self._decode(raw)
                    self._check_integrity(message)
                    event_ts = message.get("event_time") or message.get("T") or message.get("time")
                    if isinstance(event_ts, (int, float)):
                        now_ms = time.time() * 1000.0
                        raw_ts = float(event_ts)
                        ts_ms = raw_ts if raw_ts > 1e12 else raw_ts * 1000
                        self.stats.last_latency_ms = max(0.0, now_ms - ts_ms)
                    self.stats.messages += 1
                    await self.handler(message)
            finally:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

    async def _heartbeat_loop(self, ws: Any) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.heartbeat_interval)
            if self.send_heartbeat is not None:
                await self.send_heartbeat(ws)
            elif hasattr(ws, "ping"):
                await ws.ping()
            logger.debug("ws_heartbeat url={}", self.url)

    def _decode(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise DataError(
                f"Unsupported websocket payload type: {type(raw)}",
                code="WS_BAD_PAYLOAD",
            )
        if not isinstance(data, dict):
            raise DataError("Websocket payload must be an object", code="WS_BAD_PAYLOAD")
        return data

    def _check_integrity(self, message: dict[str, Any]) -> None:
        msg_id = message.get(self.message_id_key)
        if msg_id is not None:
            key = str(msg_id)
            if key in self._seen_ids:
                self.stats.duplicates += 1
                logger.warning("ws_duplicate id={}", key)
                return
            self._seen_ids.add(key)
            if len(self._seen_ids) > 50_000:
                self._seen_ids = set(list(self._seen_ids)[25_000:])

        seq = message.get(self.expected_sequence_key)
        if isinstance(seq, int):
            if self._last_sequence is not None and seq > self._last_sequence + 1:
                gap = seq - self._last_sequence - 1
                self.stats.sequence_gaps += gap
                self.stats.packet_loss_events += 1
                logger.warning(
                    "ws_sequence_gap last={} current={} missing={}",
                    self._last_sequence,
                    seq,
                    gap,
                )
            self._last_sequence = seq
