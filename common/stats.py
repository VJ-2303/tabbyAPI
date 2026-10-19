"""
Server statistics ledger.

Counts what the server has done: requests, tokens generated and processed,
cache reuse, timings, draft acceptance and finish reasons. Live values are read
from the running server when a snapshot is taken; everything else is accumulated
here and persisted to logs/stats.json so totals survive a restart.

Deliberately dependency-free: only the standard library is imported at module
level, so the ledger can be exercised by tests without a GPU, a model or the
web stack.
"""

import asyncio
import json
import os
import pathlib
import time
from collections import deque
from typing import Optional

STATS_PATH = "logs/stats.json"
DOC_VERSION = 1

# ponytail: a fixed-interval flush instead of a write per request. A hard crash
# can lose up to one interval of counters; lower it or save per generation if
# that ever matters.
SAVE_INTERVAL = 60.0

# Bounded rings, so the file stays small no matter how long the server runs
HOURLY_BUCKETS = 48
DAILY_BUCKETS = 14

# Generation rates kept for percentiles. Rates swing too much per request for a
# meaningful percentile from a handful of samples
TPS_SAMPLE_MAX = 500
TPS_PERCENTILE_MIN_SAMPLES = 5

HOUR = 3600
DAY = 24 * 3600

FINISH_MAX_CHARS = 40


def _as_int(value) -> int:
    """Counts from the backend, which reports 0, None or "Indeterminate"."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(tokens: int, seconds: float) -> Optional[float]:
    if tokens <= 0 or seconds <= 0:
        return None
    return round(tokens / seconds, 2)


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 2)


def _blank_counters() -> dict:
    return {
        "requests": 0,
        "gen_tokens": 0,
        "prompt_tokens": 0,
        "processed_tokens": 0,
        "cached_tokens": 0,
        "prompt_time": 0.0,
        "gen_time": 0.0,
        "queue_time": 0.0,
        "total_time": 0.0,
        "draft_accept": 0,
        "draft_reject": 0,
        "failed": 0,
        "cancelled": 0,
        "finish": {},
        "embedding_requests": 0,
        "embedding_tokens": 0,
        "embedding_time": 0.0,
    }


def _blank_bucket(key: str, value: int) -> dict:
    return {
        key: value,
        "requests": 0,
        "gen_tokens": 0,
        "prompt_tokens": 0,
        "processed_tokens": 0,
        "cached_tokens": 0,
        "embedding_requests": 0,
        "embedding_tokens": 0,
    }


def _merge_counters(stored) -> dict:
    """Counters from a stored file, filling in anything it doesn't have."""

    counters = _blank_counters()
    if not isinstance(stored, dict):
        return counters

    for name in counters:
        value = stored.get(name)
        if name == "finish":
            if isinstance(value, dict):
                counters[name] = {
                    str(reason): _as_int(count)
                    for reason, count in value.items()
                    if _as_int(count) > 0
                }
        elif isinstance(counters[name], float):
            counters[name] = _as_float(value)
        else:
            counters[name] = _as_int(value)

    return counters


def _merge_model(stored) -> dict:
    """A model's counters plus how often and when it was loaded."""

    entry = _merge_counters(stored) | {"loads": 0, "last_loaded": 0.0}
    if isinstance(stored, dict):
        entry["loads"] = _as_int(stored.get("loads"))
        entry["last_loaded"] = _as_float(stored.get("last_loaded"))

    return entry


class StatsLedger:
    """Counters for one server process, with the on-disk document behind them."""

    def __init__(self, path: str = STATS_PATH, clock=time.time):
        self.path = pathlib.Path(path)
        self._clock = clock
        self.session_started = self._clock()
        self.session = _blank_counters()
        self.doc = self._blank_doc()
        self._dirty = False
        # Session-only rate samples: percentiles of a persisted ring would
        # describe an arbitrary slice of history, so they aren't carried over
        self._tps_samples: deque = deque(maxlen=TPS_SAMPLE_MAX)

    # Document access

    @staticmethod
    def _blank_doc() -> dict:
        return {
            "version": DOC_VERSION,
            "started_at": None,
            "sessions": 0,
            "lifetime": _blank_counters(),
            "models": {},
            "hourly": [],
            "daily": [],
        }

    @property
    def lifetime(self) -> dict:
        return self.doc["lifetime"]

    @property
    def models(self) -> dict:
        return self.doc["models"]

    @property
    def hourly(self) -> list:
        return self.doc["hourly"]

    @property
    def daily(self) -> list:
        return self.doc["daily"]

    @property
    def dirty(self) -> bool:
        return self._dirty

    # Persistence

    async def load(self) -> str:
        """
        Read the stored document. Returns "resumed", "fresh" when there is
        nothing to read, or "corrupt" when the file can't be used, in which
        case counting starts over rather than failing to boot.
        """

        self.session_started = self._clock()
        status = "fresh"

        try:
            raw = self.path.read_text(encoding="utf8")
        except OSError:
            raw = None

        if raw is not None:
            try:
                self.doc = self._restore(json.loads(raw))
                status = "resumed"
            except (TypeError, ValueError):
                self.doc = self._blank_doc()
                status = "corrupt"

        self.doc["version"] = DOC_VERSION
        self.doc["sessions"] = _as_int(self.doc.get("sessions")) + 1
        if not self.doc.get("started_at"):
            self.doc["started_at"] = self.session_started

        return status

    @staticmethod
    def _restore(stored) -> dict:
        if not isinstance(stored, dict):
            raise TypeError("stats document is not an object")

        doc = StatsLedger._blank_doc()
        doc["started_at"] = _as_float(stored.get("started_at")) or None
        doc["sessions"] = _as_int(stored.get("sessions"))
        doc["lifetime"] = _merge_counters(stored.get("lifetime"))

        models = stored.get("models")
        if isinstance(models, dict):
            doc["models"] = {str(name): _merge_model(value) for name, value in models.items()}

        for key, cap in (("hourly", HOURLY_BUCKETS), ("daily", DAILY_BUCKETS)):
            buckets = stored.get(key)
            if isinstance(buckets, list):
                doc[key] = [bucket for bucket in buckets if isinstance(bucket, dict)][-cap:]

        return doc

    async def save(self):
        """Write the document, replacing it atomically."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.doc, separators=(",", ":"))
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(payload, encoding="utf8")
        os.replace(temporary, self.path)
        self._dirty = False

    async def save_if_dirty(self):
        if self._dirty:
            await self.save()

    async def periodic_save(self, interval: float = SAVE_INTERVAL):
        """Flush changes on a timer until the task is cancelled."""

        while True:
            await asyncio.sleep(interval)
            try:
                await self.save_if_dirty()
            except OSError as ex:
                # Losing statistics must never take the server down with them
                from common.logger import xlogger

                xlogger.warning(f"Could not write {self.path}: {ex}")

    # Recording

    def record_generation(
        self,
        metrics: Optional[dict] = None,
        *,
        model: Optional[str] = None,
        outcome: str = "finished",
    ):
        """
        Count one finished, failed or cancelled generation. `metrics` is the
        backend's finish chunk; anything it doesn't carry counts as zero, which
        is what a cancelled or failed request has.
        """

        metrics = metrics if isinstance(metrics, dict) else {}

        prompt_tokens = _as_int(metrics.get("prompt_tokens"))
        cached_tokens = _as_int(metrics.get("cached_tokens"))
        # Cache hits cost nothing to process, so only the rest is prompt work
        processed_tokens = max(prompt_tokens - cached_tokens, 0)
        gen_tokens = _as_int(metrics.get("gen_tokens"))
        gen_time = _as_float(metrics.get("gen_time"))
        prompt_time = _as_float(metrics.get("prompt_time"))

        reason = metrics.get("eos_reason") or metrics.get("finish_reason") or outcome
        finish = str(reason)[:FINISH_MAX_CHARS]

        counts = {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "processed_tokens": processed_tokens,
            "gen_tokens": gen_tokens,
            "prompt_time": prompt_time,
            "gen_time": gen_time,
            "queue_time": _as_float(metrics.get("queue_time")),
            "total_time": _as_float(metrics.get("total_time")),
            "draft_accept": _as_int(metrics.get("draft_accept")),
            "draft_reject": _as_int(metrics.get("draft_reject")),
        }

        for counters in (self.session, self.lifetime):
            counters["requests"] += 1
            for name, value in counts.items():
                counters[name] += value
            counters["finish"][finish] = counters["finish"].get(finish, 0) + 1
            if outcome != "finished":
                counters[outcome if outcome in ("failed", "cancelled") else "failed"] += 1

        if model:
            entry = self._model_entry(model)
            for name, value in counts.items():
                entry[name] += value

        if gen_time > 0 and gen_tokens > 0:
            self._tps_samples.append(gen_tokens / gen_time)

        self._roll_buckets()
        self._add_to_buckets(counts, requests=1)

        self._dirty = True

    def record_embeddings(self, *, tokens: int = 0, seconds: float = 0.0):
        """Count one embeddings request."""

        for counters in (self.session, self.lifetime):
            counters["embedding_requests"] += 1
            counters["embedding_tokens"] += _as_int(tokens)
            counters["embedding_time"] += _as_float(seconds)

        self._roll_buckets()
        self._add_to_buckets(
            {"embedding_tokens": _as_int(tokens)}, requests=1, embedding_requests=1
        )
        self._dirty = True

    def record_model_load(self, model: str):
        """Count a model load, which is when a model's counters first appear."""

        entry = self._model_entry(model)
        entry["loads"] += 1
        entry["last_loaded"] = self._clock()
        self._dirty = True

    # Internals

    def _model_entry(self, model: str) -> dict:
        return self.models.setdefault(str(model), _merge_model(None))

    def _roll_buckets(self):
        now = self._clock()
        self._roll_bucket(self.hourly, "hour", int(now // HOUR) * HOUR, HOURLY_BUCKETS)
        self._roll_bucket(self.daily, "day", int(now // DAY) * DAY, DAILY_BUCKETS)

    @staticmethod
    def _roll_bucket(buckets: list, key: str, value: int, cap: int):
        if not buckets or buckets[-1].get(key) != value:
            buckets.append(_blank_bucket(key, value))
            del buckets[:-cap]

    def _add_to_buckets(self, counts: dict, requests: int = 0, embedding_requests: int = 0):
        for bucket in (self.hourly[-1], self.daily[-1]):
            bucket["requests"] += requests
            bucket["embedding_requests"] += embedding_requests
            for name in (
                "gen_tokens",
                "prompt_tokens",
                "processed_tokens",
                "cached_tokens",
                "embedding_tokens",
            ):
                bucket[name] += _as_int(counts.get(name))

    # Snapshots

    def build_snapshot(
        self,
        *,
        live: Optional[dict] = None,
        model: Optional[dict] = None,
        gpu: Optional[list] = None,
        incidents: Optional[list] = None,
    ) -> dict:
        """Shape the counters and any gathered live values for the API."""

        now = self._clock()
        return {
            "server": {
                "version": DOC_VERSION,
                "now": now,
                "started_at": self.doc.get("started_at"),
                "sessions": _as_int(self.doc.get("sessions")),
            },
            "model": model,
            "gpu": gpu or [],
            "live": live,
            "session": self._scope_view(self.session, uptime=int(now - self.session_started)),
            "lifetime": self._scope_view(self.lifetime),
            "models": sorted(
                (
                    {"name": name, **entry}
                    for name, entry in self.models.items()
                ),
                key=lambda entry: entry["gen_tokens"],
                reverse=True,
            ),
            "history": {"hourly": self.hourly, "daily": self.daily},
            "incidents": incidents or [],
        }

    def _scope_view(self, counters: dict, uptime: Optional[int] = None) -> dict:
        view = dict(counters)
        view["finish"] = dict(counters["finish"])
        view["gen_tokens_per_sec"] = _rate(counters["gen_tokens"], counters["gen_time"])
        view["prompt_tokens_per_sec"] = _rate(
            counters["processed_tokens"], counters["prompt_time"]
        )

        # Present but null when there is too little to say, like the rates
        samples = list(self._tps_samples)
        enough = len(samples) >= TPS_PERCENTILE_MIN_SAMPLES
        view["gen_tokens_per_sec_p50"] = _percentile(samples, 0.5) if enough else None
        view["gen_tokens_per_sec_p95"] = _percentile(samples, 0.95) if enough else None

        if uptime is not None:
            view["uptime"] = uptime

        return view

    async def snapshot(self, container=None) -> dict:
        """
        A full reading of the server: live values gathered from the running
        server where it has them, counters from here. Every source is optional,
        so the page still renders while nothing is loaded. The loaded model's
        container is passed in by the endpoint, which keeps the dependency
        between this module and the model one-way.
        """

        from common import hardware
        from common.health import HealthManager

        incidents = []
        try:
            _, issues = await HealthManager.is_service_healthy()
            incidents = [
                {"time": issue.time.timestamp(), "description": issue.description}
                for issue in issues
            ]
            incidents.reverse()
        except Exception:
            pass

        return self.build_snapshot(
            live=self._live_view(container),
            model=self._model_view(container),
            gpu=hardware.gpu_memory_stats(),
            incidents=incidents,
        )

    @staticmethod
    def _live_view(container) -> Optional[dict]:
        """In-flight jobs and cache usage, or None while no model is loaded."""

        if not (container and container.loaded):
            return None

        from common.status_display import status_display

        jobs = []
        for status in status_display.jobs.values():
            jobs.append(
                {
                    "label": status.label,
                    "stage": status.stage,
                    "prompt_tokens": status.prompt_tokens,
                    "prefill_tokens": status.prefill_tokens,
                    "cached_tokens": status.cached_tokens,
                    "gen_tokens": status.gen_tokens,
                    "gen_tokens_per_sec": status.tokens_per_second(),
                }
            )

        cache = status_display.cache_stats()
        view: dict = {"jobs": jobs, "active": len(jobs)}
        if cache:
            view["queued"] = _as_int(cache.get("pending_jobs"))
            view["active"] = max(len(jobs) - view["queued"], 0)
            view["generating_tokens_per_sec"] = round(
                sum(job["gen_tokens_per_sec"] or 0 for job in jobs), 2
            ) or None
            view["cache"] = cache

        return view

    @staticmethod
    def _model_view(container) -> Optional[dict]:
        """The loaded model's own parameters, or None when nothing is loaded."""

        if not (container and container.loaded):
            return None

        try:
            card = container.model_info()
        except Exception:
            return None

        parameters = getattr(card, "parameters", None)
        meta = getattr(card, "meta", None)
        return {
            "id": getattr(card, "id", None),
            "path": str(getattr(container, "model_dir", "") or ""),
            "max_seq_len": getattr(parameters, "max_seq_len", None),
            "cache_mode": getattr(parameters, "cache_mode", None),
            "cache_size": getattr(parameters, "cache_size", None),
            "max_batch_size": getattr(parameters, "max_batch_size", None),
            "chunk_size": getattr(parameters, "chunk_size", None),
            "use_vision": getattr(parameters, "use_vision", None),
            "prompt_template": getattr(parameters, "prompt_template", None),
            "n_ctx_train": getattr(meta, "n_ctx_train", None),
            "n_vocab": getattr(meta, "n_vocab", None),
            "n_embd": getattr(meta, "n_embd", None),
            "size_bytes": getattr(meta, "size", None),
        }


# Single ledger for the process, like the health manager and status display
stats = StatsLedger()