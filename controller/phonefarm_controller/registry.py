"""In-memory worker registry; persistence is deliberately deferred to Stage 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class Worker:
    worker_id: str
    capabilities: dict[str, Any]
    status: str = "ready"
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, Any] = field(default_factory=dict)


class WorkerRegistry:
    def __init__(self, timeout_seconds: int = 15) -> None:
        self.timeout = timedelta(seconds=timeout_seconds)
        self._workers: dict[str, Worker] = {}

    def register(self, worker_id: str, capabilities: dict[str, Any]) -> Worker:
        worker = Worker(worker_id=worker_id, capabilities=capabilities)
        self._workers[worker_id] = worker
        return worker

    def heartbeat(self, worker_id: str, metrics: dict[str, Any]) -> Worker:
        worker = self._workers[worker_id]
        worker.last_heartbeat = datetime.now(timezone.utc)
        worker.metrics = metrics
        worker.status = "ready"
        return worker

    def snapshot(self, now: datetime | None = None) -> list[Worker]:
        now = now or datetime.now(timezone.utc)
        for worker in self._workers.values():
            if now - worker.last_heartbeat > self.timeout:
                worker.status = "offline"
        return list(self._workers.values())

