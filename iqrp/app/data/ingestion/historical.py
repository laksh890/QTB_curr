"""Historical paginated downloader with retries and resume checkpoints."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from iqrp.app.common.datetime_utils import ensure_utc
from iqrp.app.core.exceptions import DataError
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.models import Candle, FundingRate, OpenInterest, Trade
from iqrp.app.data.types import Timeframe, timeframe_to_timedelta, utc_to_ms


class HistoricalIngestor:
    """Paginate REST history with automatic retries and checkpoint resume."""

    def __init__(
        self,
        exchange: BaseExchange,
        *,
        page_limit: int = 1000,
        max_retries: int = 5,
        retry_delay: float = 0.5,
        retry_backoff: float = 2.0,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.exchange = exchange
        self.page_limit = page_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.checkpoint_dir = checkpoint_dir

    def _checkpoint_path(self, key: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe = key.replace("/", "_")
        return self.checkpoint_dir / f"{safe}.json"

    def load_checkpoint(self, key: str) -> dict[str, Any] | None:
        path = self._checkpoint_path(key)
        if path is None or not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return None
        return loaded

    def save_checkpoint(self, key: str, payload: dict[str, Any]) -> None:
        path = self._checkpoint_path(key)
        if path is None:
            return
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        logger.debug("checkpoint_saved key={} path={}", key, path)

    def clear_checkpoint(self, key: str) -> None:
        path = self._checkpoint_path(key)
        if path is not None and path.exists():
            path.unlink()

    async def _with_retries(self, coro_factory: Any) -> Any:
        delay = self.retry_delay
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await coro_factory()
            except DataError as exc:
                last_exc = exc
                logger.warning(
                    "download_retry attempt={} / {} error={}",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(delay)
                delay *= self.retry_backoff
        assert last_exc is not None
        raise last_exc

    async def download_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
        resume: bool = True,
    ) -> list[Candle]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        key = f"candles:{self.exchange.name}:{symbol}:{timeframe}"
        cursor = start
        if resume:
            checkpoint = self.load_checkpoint(key)
            if checkpoint and "cursor_ms" in checkpoint:
                cursor = datetime.fromtimestamp(
                    int(checkpoint["cursor_ms"]) / 1000.0, tz=start.tzinfo
                )
                logger.info("download_resume key={} cursor={}", key, cursor.isoformat())

        step = timeframe_to_timedelta(timeframe)
        collected: list[Candle] = []
        while cursor <= end:
            page_end = end
            page = await self._with_retries(
                lambda c=cursor, e=page_end: self.exchange.fetch_candles(
                    symbol, timeframe, start=c, end=e, limit=self.page_limit
                )
            )
            if not page:
                break
            # Keep candles within requested window.
            page = [c for c in page if start <= c.open_time <= end]
            if not page:
                break
            collected.extend(page)
            last_open = page[-1].open_time
            next_cursor = last_open + step
            progress = min(100.0, (utc_to_ms(last_open) - utc_to_ms(start))
                           / max(1, utc_to_ms(end) - utc_to_ms(start)) * 100.0)
            logger.info(
                "download_progress product=candles symbol={} tf={} rows={} progress={:.2f}%",
                symbol,
                timeframe,
                len(collected),
                progress,
            )
            self.save_checkpoint(key, {"cursor_ms": utc_to_ms(next_cursor), "rows": len(collected)})
            if len(page) < self.page_limit and last_open >= end:
                break
            if next_cursor <= cursor:
                # Guard against non-advancing APIs.
                next_cursor = cursor + step * max(1, len(page))
            cursor = next_cursor
            if cursor > end:
                break

        self.clear_checkpoint(key)
        # Deduplicate while preserving order.
        seen: set[datetime] = set()
        unique: list[Candle] = []
        for candle in sorted(collected, key=lambda c: c.open_time):
            if candle.open_time in seen:
                continue
            seen.add(candle.open_time)
            unique.append(candle)
        return unique

    async def download_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[Trade]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        cursor = start
        collected: list[Trade] = []
        while cursor <= end:
            page = await self._with_retries(
                lambda c=cursor: self.exchange.fetch_trades(
                    symbol, start=c, end=end, limit=self.page_limit
                )
            )
            if not page:
                break
            collected.extend(page)
            last_ts = page[-1].timestamp
            if last_ts <= cursor:
                break
            cursor = last_ts
            if len(page) < self.page_limit:
                break
        return collected

    async def download_funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[FundingRate]:
        result = await self._with_retries(
            lambda: self.exchange.fetch_funding(
                symbol, start=ensure_utc(start), end=ensure_utc(end), limit=self.page_limit
            )
        )
        return list(result)

    async def download_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[OpenInterest]:
        result = await self._with_retries(
            lambda: self.exchange.fetch_open_interest(
                symbol, start=ensure_utc(start), end=ensure_utc(end), limit=self.page_limit
            )
        )
        return list(result)
