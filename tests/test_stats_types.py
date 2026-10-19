import unittest

from common.stats import StatsLedger
from endpoints.core.types.stats import StatsResponse


def metrics(**overrides):
    base = {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "prompt_time": 0.06,
        "gen_tokens": 50,
        "gen_time": 1.2,
        "queue_time": 0.0,
        "total_time": 1.26,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
    }
    base.update(overrides)
    return base


class SnapshotSchemaTests(unittest.TestCase):
    """The ledger builds plain dicts; the endpoint promises StatsResponse."""

    def setUp(self):
        self.ledger = StatsLedger()

    def test_empty_snapshot_matches_the_response_model(self):
        response = StatsResponse.model_validate(self.ledger.build_snapshot())

        self.assertEqual(response.session.requests, 0)
        self.assertEqual(response.lifetime.requests, 0)
        self.assertEqual(response.models, [])
        self.assertIsNone(response.live)
        self.assertIsNone(response.model)

    def test_populated_snapshot_matches_the_response_model(self):
        self.ledger.record_generation(metrics(), model="llama-8b")
        self.ledger.record_model_load("llama-8b")
        self.ledger.record_embeddings(tokens=120, seconds=0.3)

        response = StatsResponse.model_validate(
            self.ledger.build_snapshot(
                live={
                    "jobs": [
                        {
                            "label": "#2 chat/completions",
                            "stage": "generating",
                            "prompt_tokens": 10,
                            "prefill_tokens": 10,
                            "cached_tokens": 0,
                            "gen_tokens": 4,
                            "gen_tokens_per_sec": 42.5,
                        }
                    ],
                    "active": 1,
                    "queued": 0,
                    "generating_tokens_per_sec": 42.5,
                    "cache": {"max_tokens": 8192, "used_tokens": 512, "hit_rate": 0.9},
                },
                model={"id": "llama-8b", "max_seq_len": 8192, "use_vision": False},
                gpu=[{"index": 0, "name": "RTX 4090", "used_bytes": 10, "total_bytes": 20,
                      "free_bytes": 10}],
                incidents=[{"time": 1.0, "description": "boom"}],
            )
        )

        self.assertEqual(response.session.requests, 1)
        self.assertEqual(response.session.processed_tokens, 100)
        self.assertEqual(response.session.embedding_tokens, 120)
        self.assertEqual(response.models[0].name, "llama-8b")
        self.assertEqual(response.models[0].loads, 1)
        self.assertEqual(response.live.jobs[0].gen_tokens_per_sec, 42.5)
        self.assertEqual(response.live.cache.max_tokens, 8192)
        self.assertEqual(response.model.max_seq_len, 8192)
        self.assertFalse(response.model.use_vision)
        self.assertEqual(response.gpu[0].used_bytes, 10)
        self.assertEqual(response.incidents[0].description, "boom")

    def test_history_buckets_match_the_response_model(self):
        self.ledger.record_generation(metrics())

        response = StatsResponse.model_validate(self.ledger.build_snapshot())

        self.assertEqual(len(response.history.hourly), 1)
        self.assertEqual(response.history.hourly[0].gen_tokens, 50)
        self.assertEqual(response.history.daily[0].requests, 1)

    def test_cache_keys_the_backend_adds_are_ignored(self):
        # The cache block is what the backend reports; a key this API doesn't
        # know must not break the response
        response = StatsResponse.model_validate(
            self.ledger.build_snapshot(
                live={
                    "jobs": [],
                    "active": 0,
                    "cache": {"max_tokens": 10, "used_tokens": 5, "tokens_per_page": 32},
                }
            )
        )

        self.assertEqual(response.live.cache.max_tokens, 10)
        self.assertFalse(hasattr(response.live.cache, "tokens_per_page"))