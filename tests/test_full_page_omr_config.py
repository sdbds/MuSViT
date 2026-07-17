import json
import unittest

from experiments.full_page_omr.config.ExperimentConfigWrapper import (
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


if __name__ == "__main__":
    unittest.main()
