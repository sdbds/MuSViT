import json
import unittest
from copy import deepcopy
from pathlib import Path

from experiments.full_page_omr.config.ExperimentConfigWrapper import (
    Data,
    PDMXData,
    experiment_config_from_dict,
    experiment_config_to_dict,
)


class FullPageOMRConfigTests(unittest.TestCase):
    @staticmethod
    def _config(**overrides):
        data = {
            "data_path": "example/dataset",
            "batch_size": 1,
            "vocab_name": "Example_BeKern",
            "num_workers": 24,
            "tokenization_mode": "bekern",
            "reduce_ratio": 0.5,
            "skip_steps": 120000,
        }
        data.update(overrides)
        return {"data": data}

    @staticmethod
    def _pdmx_config(**overrides):
        data = {
            "type": "pdmx_webdataset",
            "dataset_id": "tobiashornbogen/page-omr-pdmx-renders",
            "dataset_revision": "7da3ae5237963e57a8fe1c6ee375b1f10af34a09",
            "dataset_manifest": "config/Page_OMR_PDMX/dataset-manifest.v1.json",
            "vocab_manifest": "vocab/FullPageOMR_BeKern_v1.json",
            "renderer_weights": {"verovio": 0.5, "mscore": 0.5},
            "batch_size": 1,
            "num_workers": 8,
            "tokenization_mode": "bekern",
            "steps_per_epoch": 10000,
            "shuffle_buffer": 2048,
            "seed": 3407,
            "runtime_augmentation": False,
        }
        data.update(overrides)
        return {"data": data}

    def test_skip_steps_is_preserved_when_parsing_config(self):
        config = experiment_config_from_dict(
            {
                "data": {
                    "data_path": "example/dataset",
                    "batch_size": 1,
                    "vocab_name": "Example",
                    "num_workers": 0,
                    "tokenization_mode": "bekern",
                    "reduce_ratio": 0.5,
                    "skip_steps": 120000,
                }
            }
        )

        self.assertEqual(config.data.skip_steps, 120000)

    def test_config_round_trip_preserves_nonzero_skip_steps(self):
        config = experiment_config_from_dict(self._config())

        serialized = experiment_config_to_dict(config)
        restored = experiment_config_from_dict(json.loads(json.dumps(serialized)))

        self.assertEqual(restored, config)

    def test_batch_size_other_than_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            experiment_config_from_dict(self._config(batch_size=2))

    def test_negative_worker_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_workers"):
            experiment_config_from_dict(self._config(num_workers=-1))

    def test_unknown_tokenization_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tokenization_mode"):
            experiment_config_from_dict(self._config(tokenization_mode="standard"))

    def test_boolean_is_not_accepted_as_an_integer(self):
        with self.assertRaisesRegex(TypeError, "skip_steps"):
            experiment_config_from_dict(self._config(skip_steps=True))

    def test_fp_grandstaff_config_uses_independent_dataset_and_vocab(self):
        experiment_root = (
            Path(__file__).resolve().parents[1] / "experiments" / "full_page_omr"
        )
        config_path = experiment_root / "config" / "FP_GrandStaff" / "finetuning.json"
        config = experiment_config_from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )

        self.assertEqual(config.data.data_path, "PRAIG/fp-grandstaff")
        self.assertEqual(config.data.vocab_name, "FP_GrandStaff_BeKern")
        self.assertEqual(config.data.tokenization_mode, "bekern")
        self.assertEqual(config.data.batch_size, 1)
        self.assertEqual(config.data.num_workers, 24)
        self.assertEqual(config.data.reduce_ratio, 1.0)

        vocab_root = experiment_root / "vocab"
        self.assertTrue(
            (vocab_root / f"{config.data.vocab_name}w2i.npy").is_file()
        )
        self.assertTrue(
            (vocab_root / f"{config.data.vocab_name}i2w.npy").is_file()
        )

    def test_missing_type_keeps_legacy_data_contract(self):
        config = experiment_config_from_dict(self._config())

        self.assertIsInstance(config.data, Data)

    def test_pdmx_config_parses_and_round_trips(self):
        payload = self._pdmx_config()

        config = experiment_config_from_dict(payload)
        restored = experiment_config_from_dict(
            json.loads(json.dumps(experiment_config_to_dict(config)))
        )

        self.assertIsInstance(config.data, PDMXData)
        self.assertEqual(config.data.skip_steps, 0)
        self.assertEqual(config.data.reduce_ratio, 1.0)
        self.assertEqual(
            config.data.renderer_weights,
            {"mscore": 0.5, "verovio": 0.5},
        )
        self.assertEqual(restored, config)

    def test_pdmx_rejects_arrow_only_fields(self):
        for field, value in (("reduce_ratio", 1.0), ("skip_steps", 0)):
            with self.subTest(field=field):
                payload = deepcopy(self._pdmx_config())
                payload["data"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    experiment_config_from_dict(payload)

    def test_pdmx_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            experiment_config_from_dict(
                self._pdmx_config(allow_download=True)
            )

    def test_pdmx_rejects_unpinned_revision(self):
        with self.assertRaisesRegex(ValueError, "dataset_revision"):
            experiment_config_from_dict(
                self._pdmx_config(dataset_revision="main")
            )

    def test_pdmx_rejects_renderer_weight_changes(self):
        with self.assertRaisesRegex(ValueError, "renderer_weights"):
            experiment_config_from_dict(
                self._pdmx_config(
                    renderer_weights={"verovio": 0.75, "mscore": 0.25}
                )
            )

    def test_pdmx_rejects_runtime_augmentation(self):
        with self.assertRaisesRegex(ValueError, "runtime_augmentation"):
            experiment_config_from_dict(
                self._pdmx_config(runtime_augmentation=True)
            )

    def test_new_pdmx_and_downstream_configs_parse(self):
        config_root = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "full_page_omr"
            / "config"
        )
        cases = (
            (
                config_root / "Page_OMR_PDMX" / "pretraining.json",
                PDMXData,
            ),
            (
                config_root / "Polish_Scores" / "pdmx_finetuning.json",
                Data,
            ),
            (
                config_root / "Mozarteum" / "pdmx_finetuning.json",
                Data,
            ),
        )
        for path, expected_type in cases:
            with self.subTest(path=path):
                payload = json.loads(path.read_text(encoding="utf-8"))
                config = experiment_config_from_dict(payload)
                self.assertIsInstance(config.data, expected_type)
                if isinstance(config.data, PDMXData):
                    self.assertEqual(
                        config.data.vocab_manifest,
                        "vocab/FullPageOMR_BeKern_v1.json",
                    )
                else:
                    self.assertEqual(
                        config.data.vocab_name,
                        "FullPageOMR_BeKern_v1",
                    )


if __name__ == "__main__":
    unittest.main()
