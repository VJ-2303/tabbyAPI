import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common import hardware


def fake_torch(devices, error=None):
    """Stands in for the torch module; mem_get_info returns (free, total)."""

    def mem_get_info(index):
        if error is not None:
            raise error
        device = devices[index]
        return (device["total"] - device["used"], device["total"])

    cuda = SimpleNamespace(
        is_available=lambda: bool(devices),
        device_count=lambda: len(devices),
        get_device_name=lambda index: devices[index]["name"],
        mem_get_info=mem_get_info,
    )
    return SimpleNamespace(cuda=cuda)


class GpuMemoryStatsTests(unittest.TestCase):
    def test_reports_used_and_total_per_gpu(self):
        devices = [
            {"name": "RTX 4090", "used": 8 * 10**9, "total": 24 * 10**9},
            {"name": "RTX 4090", "used": 2 * 10**9, "total": 24 * 10**9},
        ]

        with patch("common.hardware._torch", return_value=fake_torch(devices)):
            stats = hardware.gpu_memory_stats()

        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["index"], 0)
        self.assertEqual(stats[0]["name"], "RTX 4090")
        self.assertEqual(stats[1]["used_bytes"], 2 * 10**9)
        self.assertEqual(stats[1]["total_bytes"], 24 * 10**9)
        self.assertEqual(stats[1]["free_bytes"], 22 * 10**9)

    def test_returns_none_without_a_gpu(self):
        with patch("common.hardware._torch", return_value=fake_torch([])):
            self.assertIsNone(hardware.gpu_memory_stats())

    def test_returns_none_when_the_driver_refuses_to_report(self):
        devices = [{"name": "RTX 4090", "used": 1, "total": 2}]

        with patch(
            "common.hardware._torch",
            return_value=fake_torch(devices, error=RuntimeError("no memory info")),
        ):
            self.assertIsNone(hardware.gpu_memory_stats())

    def test_returns_none_when_torch_is_not_installed(self):
        with patch("common.hardware._torch", side_effect=ImportError("no torch")):
            self.assertIsNone(hardware.gpu_memory_stats())