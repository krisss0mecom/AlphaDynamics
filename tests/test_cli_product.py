import json
import tempfile
import unittest
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import alphadynamics_cli as cli
import timewarp_comparison as twc


class ProductCliTest(unittest.TestCase):
    def test_report_summarizers_read_shipped_artifacts(self):
        nll = cli.summarize_nll(ROOT / "results" / "mdcath_aligned20_4000step_cpu.json")
        self.assertEqual(nll["domains"], 20)
        self.assertEqual(nll["wins"], 20)
        self.assertGreater(nll["ratio"], 1.0)
        self.assertGreater(nll["ad_params"], 0)

        rollout = cli.summarize_rollout(
            ROOT / "results" / "ramachandran_aligned3_4000step_gpu.json"
        )
        self.assertEqual(rollout["domains"], 3)
        self.assertGreater(rollout["jsd"], 0.0)

    def test_strong_baseline_summarizer_handles_product_payload(self):
        payload = {
            "run": {"steps": 1},
            "results": [
                {
                    "models": {
                        "MLP_abs": {"nll": 10.0},
                        "MLP_residual": {"nll": 8.0},
                        "PhaseFlow_t1": {"nll": 7.0},
                        "PhaseFlow_t4": {"nll": 9.0},
                    }
                },
                {
                    "models": {
                        "MLP_abs": {"nll": 12.0},
                        "MLP_residual": {"nll": 6.0},
                        "PhaseFlow_t1": {"nll": 7.0},
                        "PhaseFlow_t4": {"nll": 5.0},
                    }
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strong.json"
            path.write_text(json.dumps(payload))
            row = cli.summarize_strong_baseline(path)

        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["pf_res_wins"], 2)
        self.assertEqual(row["pf_abs_wins"], 2)
        self.assertEqual(row["res_abs_wins"], 2)

    def test_temporal_baseline_summarizer_handles_product_payload(self):
        payload = {
            "run": {"window": 8},
            "results": [
                {
                    "models": {
                        "MLP_abs": {"nll": 12.0},
                        "TemporalGRU": {"nll": 9.0},
                        "PhaseFlow_t4": {"nll": 7.0},
                    }
                },
                {
                    "models": {
                        "MLP_abs": {"nll": 8.0},
                        "TemporalGRU": {"nll": 6.0},
                        "PhaseFlow_t4": {"nll": 6.5},
                    }
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporal.json"
            path.write_text(json.dumps(payload))
            row = cli.summarize_temporal_baseline(path)

        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["pf_gru_wins"], 1)
        self.assertEqual(row["pf_abs_wins"], 2)
        self.assertEqual(row["gru_abs_wins"], 2)

    def test_timewarp_path_helpers(self):
        self.assertEqual(
            twc.arrays_path("4AA-large", "test", "AAEW"),
            "4AA-large/test/AAEW-traj-arrays.npz",
        )
        self.assertEqual(
            twc.pdb_path("4AA-large", "test", "AAEW"),
            "4AA-large/test/AAEW-traj-state0.pdb",
        )
        self.assertEqual(twc.domain_id_from_arrays_path("4AA-large/test/AAEW-traj-arrays.npz"), "AAEW")


if __name__ == "__main__":
    unittest.main()
