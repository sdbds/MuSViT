import json
import tempfile
import unittest
from pathlib import Path

from experiments.full_page_omr import report_metric_migration as migration


I2W = {
    0: "<pad>",
    1: "<bos>",
    2: "<eos>",
    3: "note",
}


class MetricMigrationReportTests(unittest.TestCase):
    def test_report_compares_v2_and_exact_legacy_target_semantics_per_page(self):
        prediction_ids = [(1, 3, 2)] * 10
        target_ids = [(1, 3, 2)] * 10

        report = migration.build_metric_migration_report(
            prediction_ids=prediction_ids,
            target_ids=target_ids,
            i2w=I2W,
            maxlen=4,
            checkpoint_identity={"sha256": "checkpoint"},
            dataset_identity={"revision": "dataset"},
            generation_seconds=1.25,
            environment={"resolved_attention_backend": "eager"},
        )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["rows"], list(range(10)))
        self.assertEqual(report["aggregate"]["v2"]["CER"], 0.0)
        self.assertEqual(report["aggregate"]["legacy"]["CER"], 100.0)
        self.assertEqual(
            report["aggregate"]["delta_v2_minus_legacy"]["CER"],
            -100.0,
        )
        self.assertEqual(len(report["pages"]), 10)
        self.assertEqual(report["pages"][0]["legacy_target_suffix"], "<eos>")
        self.assertTrue(report["pages"][0]["terminated_by_eos"])
        self.assertFalse(report["pages"][0]["truncated"])

    def test_report_requires_the_locked_ten_page_validation_split(self):
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            migration.build_metric_migration_report(
                prediction_ids=[(1, 3, 2)],
                target_ids=[(1, 3, 2)],
                i2w=I2W,
                maxlen=4,
                checkpoint_identity={},
                dataset_identity={},
                generation_seconds=1.0,
                environment={},
            )

    def test_report_writer_archives_structured_json(self):
        payload = {"status": "completed", "pages": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "migration.json"
            migration.write_report(path, payload)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
