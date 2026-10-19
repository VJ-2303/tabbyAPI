import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common.stats import StatsLedger


class FakeJob:
    def __init__(self, label="#1 chat/completions", gen_tokens=20, rate=42.5):
        self.label = label
        self.stage = "generating"
        self.prompt_tokens = 100
        self.prefill_tokens = 100
        self.cached_tokens = 0
        self.gen_tokens = gen_tokens
        self._rate = rate

    def tokens_per_second(self):
        return self._rate


class FakeDisplay:
    """Stands in for the terminal status line's job tracking."""

    def __init__(self, jobs=(), cache=None):
        self.jobs = {str(index): job for index, job in enumerate(jobs)}
        self._cache = cache

    def cache_stats(self):
        return self._cache


def loaded_container(card=None, error=None):
    def model_info():
        if error is not None:
            raise error
        return card

    return SimpleNamespace(loaded=True, model_dir=Path("/models/llama-8b"), model_info=model_info)


def model_card():
    parameters = SimpleNamespace(
        max_seq_len=8192, cache_mode="Q6", max_batch_size=2, chunk_size=2048, use_vision=False
    )
    meta = SimpleNamespace(n_ctx_train=8192, n_vocab=128000, n_embd=4096, size=4_000_000_000)
    return SimpleNamespace(id="llama-8b", parameters=parameters, meta=meta)


class LiveViewTests(unittest.TestCase):
    def setUp(self):
        self.ledger = StatsLedger()

    def live_view(self, display, container=None):
        container = container if container is not None else loaded_container(model_card())
        module = SimpleNamespace(status_display=display)
        with patch.dict(sys.modules, {"common.status_display": module}):
            return self.ledger._live_view(container)

    def test_no_model_loaded_reports_nothing(self):
        self.assertIsNone(self.ledger._live_view(None))
        self.assertIsNone(self.ledger._live_view(SimpleNamespace(loaded=False)))

    def test_jobs_and_cache_are_reported(self):
        view = self.live_view(
            FakeDisplay(jobs=[FakeJob(), FakeJob(rate=None)], cache={"pending_jobs": 1}),
        )

        self.assertEqual(view["active"], 1)
        self.assertEqual(view["queued"], 1)
        self.assertEqual(view["generating_tokens_per_sec"], 42.5)
        self.assertEqual(view["cache"], {"pending_jobs": 1})
        self.assertEqual(view["jobs"][0]["label"], "#1 chat/completions")
        self.assertEqual(view["jobs"][0]["gen_tokens"], 20)
        self.assertEqual(view["jobs"][1]["gen_tokens_per_sec"], None)

    def test_backend_without_cache_stats_still_reports_jobs(self):
        view = self.live_view(FakeDisplay(jobs=[FakeJob()]))

        self.assertEqual(view["active"], 1)
        self.assertNotIn("queued", view)
        self.assertNotIn("cache", view)
        self.assertNotIn("generating_tokens_per_sec", view)

    def test_idle_engine_reports_no_rate(self):
        view = self.live_view(FakeDisplay(cache={"pending_jobs": 0}))

        self.assertEqual(view["active"], 0)
        self.assertIsNone(view["generating_tokens_per_sec"])


class ModelViewTests(unittest.TestCase):
    def setUp(self):
        self.ledger = StatsLedger()

    def test_no_model_loaded_reports_nothing(self):
        self.assertIsNone(self.ledger._model_view(None))
        self.assertIsNone(self.ledger._model_view(SimpleNamespace(loaded=False)))

    def test_loaded_model_parameters_are_reported(self):
        view = self.ledger._model_view(loaded_container(model_card()))

        self.assertEqual(view["id"], "llama-8b")
        self.assertEqual(view["max_seq_len"], 8192)
        self.assertEqual(view["cache_mode"], "Q6")
        self.assertEqual(view["use_vision"], False)
        self.assertEqual(view["n_ctx_train"], 8192)
        self.assertEqual(view["size_bytes"], 4_000_000_000)
        self.assertEqual(view["path"], "/models/llama-8b")

    def test_a_model_that_cannot_describe_itself_reports_nothing(self):
        container = loaded_container(error=RuntimeError("unloaded mid-request"))

        self.assertIsNone(self.ledger._model_view(container))

    def test_missing_fields_are_null(self):
        container = loaded_container(SimpleNamespace(id="bare"))

        view = self.ledger._model_view(container)

        self.assertEqual(view["id"], "bare")
        self.assertIsNone(view["max_seq_len"])
        self.assertIsNone(view["n_vocab"])