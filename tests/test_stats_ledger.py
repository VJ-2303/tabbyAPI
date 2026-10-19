import json
import pathlib
import tempfile
import unittest

from common.stats import (
    DAILY_BUCKETS,
    HOURLY_BUCKETS,
    StatsLedger,
)

DAY = 24 * 60 * 60


class FakeClock:
    """Wall clock the ledger reads, so bucket rollovers can be tested."""

    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    def __call__(self) -> float:
        return self.timestamp


def metrics(**overrides):
    """A finish chunk as exllamav3 reports it for a completed generation."""

    base = {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "prompt_time": 0.06,
        "prompt_tokens_per_sec": 1666.67,
        "gen_tokens": 50,
        "gen_time": 1.2,
        "gen_tokens_per_sec": 41.7,
        "total_time": 1.26,
        "queue_time": 0.0,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
    }
    base.update(overrides)
    return base


class LedgerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "logs" / "stats.json"
        self.clock = FakeClock(1_700_000_000.0)
        self.ledger = StatsLedger(self.path, clock=self.clock)
        await self.ledger.load()


class RecordingTests(LedgerTestCase):
    async def test_records_counts_and_times_in_both_scopes(self):
        self.ledger.record_generation(metrics())

        for counters in (self.ledger.session, self.ledger.lifetime):
            self.assertEqual(counters["requests"], 1)
            self.assertEqual(counters["prompt_tokens"], 1000)
            self.assertEqual(counters["gen_tokens"], 50)
            self.assertEqual(counters["cached_tokens"], 900)
            self.assertAlmostEqual(counters["gen_time"], 1.2)
            self.assertAlmostEqual(counters["prompt_time"], 0.06)
            self.assertAlmostEqual(counters["queue_time"], 0.0)
            self.assertAlmostEqual(counters["total_time"], 1.26)

    async def test_cached_tokens_are_not_counted_as_processed(self):
        self.ledger.record_generation(metrics())

        self.assertEqual(self.ledger.lifetime["processed_tokens"], 100)

    async def test_finish_reason_is_counted(self):
        self.ledger.record_generation(metrics())
        self.ledger.record_generation(metrics(eos_reason="max_new_tokens"))

        self.assertEqual(
            self.ledger.lifetime["finish"], {"stop_token": 1, "max_new_tokens": 1}
        )

    async def test_draft_tokens_accumulate(self):
        self.ledger.record_generation(metrics(draft_accept=40, draft_reject=8))
        self.ledger.record_generation(metrics(draft_accept=10, draft_reject=2))

        self.assertEqual(self.ledger.lifetime["draft_accept"], 50)
        self.assertEqual(self.ledger.lifetime["draft_reject"], 10)

    async def test_failed_generation_counts_the_request_without_tokens(self):
        self.ledger.record_generation(None, outcome="failed")

        self.assertEqual(self.ledger.lifetime["requests"], 1)
        self.assertEqual(self.ledger.lifetime["failed"], 1)
        self.assertEqual(self.ledger.lifetime["gen_tokens"], 0)
        self.assertEqual(self.ledger.session["failed"], 1)

    async def test_cancelled_generation_is_counted_separately(self):
        self.ledger.record_generation(metrics(gen_tokens=5), outcome="cancelled")

        self.assertEqual(self.ledger.lifetime["cancelled"], 1)
        self.assertEqual(self.ledger.lifetime["failed"], 0)
        # Whatever was generated before the cancel still counts
        self.assertEqual(self.ledger.lifetime["gen_tokens"], 5)

    async def test_per_model_totals_are_separate(self):
        self.ledger.record_generation(metrics(), model="llama-8b")
        self.ledger.record_generation(metrics(gen_tokens=10), model="qwen-4b")
        self.ledger.record_generation(metrics(gen_tokens=1), model="llama-8b")

        self.assertEqual(self.ledger.models["llama-8b"]["gen_tokens"], 51)
        self.assertEqual(self.ledger.models["qwen-4b"]["gen_tokens"], 10)

    async def test_model_load_is_counted(self):
        self.ledger.record_model_load("llama-8b")
        self.clock.timestamp += 60
        self.ledger.record_model_load("llama-8b")

        entry = self.ledger.models["llama-8b"]
        self.assertEqual(entry["loads"], 2)
        self.assertEqual(entry["last_loaded"], self.clock.timestamp)

    async def test_embeddings_are_recorded(self):
        self.ledger.record_embeddings(tokens=120, seconds=0.4)

        self.assertEqual(self.ledger.lifetime["embedding_requests"], 1)
        self.assertEqual(self.ledger.lifetime["embedding_tokens"], 120)
        self.assertAlmostEqual(self.ledger.lifetime["embedding_time"], 0.4)

    async def test_unknown_finish_reason_is_labelled(self):
        self.ledger.record_generation({"finish_reason": "length", "gen_tokens": 3})

        self.assertEqual(self.ledger.lifetime["finish"], {"length": 1})

    async def test_negative_cache_count_is_clamped(self):
        # A backend could report more cached tokens than the prompt held
        self.ledger.record_generation(metrics(prompt_tokens=10, cached_tokens=99))

        self.assertEqual(self.ledger.lifetime["processed_tokens"], 0)


class BucketTests(LedgerTestCase):
    async def test_hourly_bucket_rolls_over_on_the_hour(self):
        self.ledger.record_generation(metrics())
        self.clock.timestamp += 3600
        self.ledger.record_generation(metrics(gen_tokens=7))

        self.assertEqual(len(self.ledger.hourly), 2)
        self.assertEqual(self.ledger.hourly[0]["requests"], 1)
        self.assertEqual(self.ledger.hourly[1]["gen_tokens"], 7)

    async def test_same_hour_reuses_one_bucket(self):
        self.ledger.record_generation(metrics())
        self.clock.timestamp += 600
        self.ledger.record_generation(metrics())

        self.assertEqual(len(self.ledger.hourly), 1)
        self.assertEqual(self.ledger.hourly[0]["requests"], 2)
        self.assertEqual(self.ledger.hourly[0]["processed_tokens"], 200)

    async def test_daily_bucket_rolls_over_on_the_day(self):
        self.ledger.record_generation(metrics())
        self.clock.timestamp += DAY
        self.ledger.record_generation(metrics())

        self.assertEqual(len(self.ledger.daily), 2)

    async def test_hourly_and_daily_buckets_are_capped(self):
        for _ in range(HOURLY_BUCKETS + 4):
            self.clock.timestamp += 3600
            self.ledger.record_generation(metrics())

        for _ in range(DAILY_BUCKETS + 4):
            self.clock.timestamp += DAY
            self.ledger.record_generation(metrics())

        self.assertEqual(len(self.ledger.hourly), HOURLY_BUCKETS)
        self.assertEqual(len(self.ledger.daily), DAILY_BUCKETS)


class PersistenceTests(LedgerTestCase):
    async def test_missing_file_loads_fresh(self):
        ledger = StatsLedger(self.path, clock=self.clock)

        self.assertEqual(await ledger.load(), "fresh")
        self.assertEqual(ledger.lifetime["requests"], 0)
        self.assertEqual(ledger.doc["sessions"], 1)

    async def test_corrupt_file_loads_fresh_without_raising(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json at all", encoding="utf8")

        self.assertEqual(await self.ledger.load(), "corrupt")
        self.assertEqual(self.ledger.lifetime["gen_tokens"], 0)

    async def test_save_and_load_resumes_lifetime_counters(self):
        self.ledger.record_generation(metrics(), model="llama-8b")
        self.ledger.record_model_load("llama-8b")
        await self.ledger.save()

        clock = FakeClock(self.clock.timestamp + 120)
        restored = StatsLedger(self.path, clock=clock)

        self.assertEqual(await restored.load(), "resumed")
        self.assertEqual(restored.lifetime["requests"], 1)
        self.assertEqual(restored.lifetime["gen_tokens"], 50)
        self.assertEqual(restored.models["llama-8b"]["loads"], 1)
        self.assertEqual(len(restored.hourly), 1)
        # The resumed process starts with a clean session scope
        self.assertEqual(restored.session["requests"], 0)
        self.assertEqual(restored.doc["sessions"], 2)

    async def test_save_writes_only_the_stats_file(self):
        self.ledger.record_generation(metrics())
        await self.ledger.save()

        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["stats.json"])

    async def test_save_if_dirty_skips_a_clean_ledger(self):
        self.assertFalse(self.ledger.dirty)
        await self.ledger.save_if_dirty()
        self.assertFalse(self.path.exists())

        self.ledger.record_generation(metrics())
        self.assertTrue(self.ledger.dirty)

        await self.ledger.save_if_dirty()
        self.assertFalse(self.ledger.dirty)
        self.assertTrue(self.path.exists())

    async def test_started_at_survives_a_restart(self):
        self.ledger.record_generation(metrics())
        await self.ledger.save()

        restored = StatsLedger(self.path, clock=FakeClock(self.clock.timestamp + 60))
        await restored.load()

        self.assertEqual(restored.doc["started_at"], 1_700_000_000.0)
        self.assertEqual(restored.session_started, self.clock.timestamp + 60)


class SnapshotTests(LedgerTestCase):
    async def test_snapshot_without_a_model_or_live_data(self):
        snapshot = self.ledger.build_snapshot()

        self.assertIn("server", snapshot)
        self.assertIn("session", snapshot)
        self.assertIn("lifetime", snapshot)
        self.assertIsNone(snapshot["model"])
        self.assertIsNone(snapshot["live"])
        self.assertEqual(snapshot["incidents"], [])
        self.assertEqual(snapshot["history"]["hourly"], [])

    async def test_snapshot_carries_live_model_and_incident_data(self):
        live = {"active": 2, "generating_tps": 55.5}
        model = {"id": "llama-8b"}
        incidents = [{"time": 1.0, "description": "boom"}]

        snapshot = self.ledger.build_snapshot(
            live=live, model=model, incidents=incidents
        )

        self.assertEqual(snapshot["live"], live)
        self.assertEqual(snapshot["model"], model)
        self.assertEqual(snapshot["incidents"], incidents)

    async def test_snapshot_reports_uptime_and_averages(self):
        self.clock.timestamp += 90
        self.ledger.record_generation(metrics())

        session = self.ledger.build_snapshot()["session"]

        self.assertEqual(session["uptime"], 90)
        self.assertAlmostEqual(session["gen_tokens_per_sec"], 50 / 1.2, places=2)
        self.assertAlmostEqual(session["prompt_tokens_per_sec"], 100 / 0.06, places=2)

    async def test_rates_are_none_without_generation_time(self):
        self.ledger.record_generation({"gen_tokens": 0, "gen_time": 0})

        session = self.ledger.build_snapshot()["session"]

        self.assertIsNone(session["gen_tokens_per_sec"])
        self.assertIsNone(session["prompt_tokens_per_sec"])

    async def test_percentiles_need_enough_samples(self):
        for seconds in range(1, 4):
            self.ledger.record_generation(metrics(gen_tokens=100, gen_time=seconds))

        session = self.ledger.build_snapshot()["session"]
        self.assertIsNone(session["gen_tokens_per_sec_p50"])

        self.ledger.record_generation(metrics(gen_tokens=100, gen_time=4))
        self.ledger.record_generation(metrics(gen_tokens=100, gen_time=5))

        session = self.ledger.build_snapshot()["session"]
        self.assertGreater(session["gen_tokens_per_sec_p95"], session["gen_tokens_per_sec_p50"])
        self.assertAlmostEqual(session["gen_tokens_per_sec_p50"], 33.33, places=1)

    async def test_models_are_ranked_by_generated_tokens(self):
        self.ledger.record_generation(metrics(gen_tokens=5), model="small")
        self.ledger.record_generation(metrics(gen_tokens=500), model="big")

        models = self.ledger.build_snapshot()["models"]

        self.assertEqual([entry["name"] for entry in models], ["big", "small"])

    async def test_snapshot_counts_are_json_serializable(self):
        self.ledger.record_generation(metrics(), model="llama-8b")
        self.ledger.record_embeddings(tokens=5, seconds=0.1)

        json.dumps(self.ledger.build_snapshot())
        json.dumps(self.ledger.doc)