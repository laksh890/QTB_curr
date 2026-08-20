"""Async multi-symbol / multi-timeframe download scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from iqrp.app.data.types import Timeframe

JobFunc = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class DownloadJob:
    symbol: str
    timeframe: str


class IngestionScheduler:
    """Run download jobs with bounded concurrency."""

    def __init__(self, *, concurrency: int = 4) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency

    def build_jobs(
        self,
        symbols: Sequence[str],
        timeframes: Sequence[Timeframe | str],
    ) -> list[DownloadJob]:
        return [
            DownloadJob(symbol=symbol, timeframe=str(timeframe))
            for symbol in symbols
            for timeframe in timeframes
        ]

    async def run(
        self,
        jobs: Sequence[DownloadJob],
        worker: JobFunc,
    ) -> None:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _run_one(job: DownloadJob) -> None:
            async with semaphore:
                logger.info("scheduler_start symbol={} tf={}", job.symbol, job.timeframe)
                await worker(job.symbol, job.timeframe)
                logger.info("scheduler_done symbol={} tf={}", job.symbol, job.timeframe)

        await asyncio.gather(*(_run_one(job) for job in jobs))

    async def run_window(
        self,
        *,
        symbols: Sequence[str],
        timeframes: Sequence[Timeframe | str],
        start: datetime,
        end: datetime,
        worker: Callable[[str, str, datetime, datetime], Awaitable[None]],
    ) -> None:
        async def adapted(symbol: str, timeframe: str) -> None:
            await worker(symbol, timeframe, start, end)

        jobs = self.build_jobs(symbols, timeframes)
        await self.run(jobs, adapted)
