import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from ranker import (
    build_recommendation, contiguous_tail, flip_rate, percentiles,
    price_efficiency, same_time_relative_volume,
)


class FactorTests(unittest.TestCase):
    def test_efficiency_distinguishes_trend_from_chop(self):
        self.assertEqual(price_efficiency([100, 101, 102, 103]), 1.0)
        self.assertLess(price_efficiency([100, 102, 99, 101]), 0.2)

    def test_flip_rate(self):
        self.assertEqual(flip_rate([100, 101, 102, 103]), 0.0)
        self.assertEqual(flip_rate([100, 101, 100, 101]), 1.0)

    def test_lunch_break_is_not_bridged(self):
        rows = [
            {"ts": 1000.0}, {"ts": 1300.0}, {"ts": 1600.0},
            {"ts": 6000.0}, {"ts": 6300.0},
        ]
        self.assertEqual(len(contiguous_tail(rows)), 2)

    def test_same_time_volume_uses_same_cutoff(self):
        today = [{"minute": 545.0, "volume": 100.0}, {"minute": 550.0, "volume": 100.0}]
        previous = [[
            {"minute": 545.0, "volume": 50.0}, {"minute": 550.0, "volume": 50.0},
            {"minute": 900.0, "volume": 900.0},
        ]]
        self.assertEqual(same_time_relative_volume(today, previous), 2.0)

    def test_percentiles_handle_ties(self):
        self.assertEqual(percentiles([1.0, 2.0, 2.0, 4.0]), [0.0, 0.5, 0.5, 1.0])

    def test_alert_requires_all_validation_gates(self):
        row = {
            "code": "9432", "name": "NTT", "price": 160.0,
            "model_score": 0.8, "realized_vol": 0.5, "window_minutes": 60,
        }
        model = {
            "paper_ready": False, "threshold": 0.7,
            "validation": {"avg_net_pct": 0.2},
        }
        now = datetime(2026, 8, 14, 10, 5, tzinfo=ZoneInfo("Asia/Tokyo"))
        self.assertFalse(build_recommendation([row], model, now)["announce"])

    def test_alert_contains_stock_amount_and_time(self):
        row = {
            "code": "9432", "name": "NTT", "price": 160.0,
            "model_score": 0.8, "realized_vol": 0.5, "window_minutes": 60,
        }
        model = {
            "paper_ready": True, "threshold": 0.7,
            "validation": {"avg_net_pct": 0.2},
        }
        now = datetime(2026, 8, 14, 10, 5, tzinfo=ZoneInfo("Asia/Tokyo"))
        result = build_recommendation([row], model, now)
        self.assertTrue(result["announce"])
        self.assertEqual(result["candidate"]["amount_yen"], 16_000)
        self.assertEqual(result["candidate"]["time_window"], "10:05〜10:20")


if __name__ == "__main__":
    unittest.main()
