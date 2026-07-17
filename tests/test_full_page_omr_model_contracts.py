import unittest
from types import SimpleNamespace

import torch

from experiments.full_page_omr.smt_foundation import modeling_smt
from experiments.full_page_omr.smt_foundation.modeling_smt import (
    PositionalEncoding2D,
    SMTFoundationModelForCausalLM,
)


class PositionalEncodingTests(unittest.TestCase):
    def test_encoding_is_built_lazily_for_the_actual_feature_grid(self):
        encoding = PositionalEncoding2D(256)

        self.assertIsNone(encoding.pe)
        features = torch.zeros((1, 256, 64, 64), dtype=torch.float32)
        result = encoding(features)

        self.assertEqual(result.shape, features.shape)
        self.assertEqual(encoding.pe.shape, (1, 256, 64, 64))
        self.assertEqual(encoding.pe.numel() * encoding.pe.element_size(), 4 * 1024 * 1024)
        self.assertNotIn("pe", encoding.state_dict())

    def test_encoding_cache_tracks_feature_dtype_and_shape(self):
        encoding = PositionalEncoding2D(4)

        encoding(torch.zeros((1, 4, 2, 3), dtype=torch.float32))
        first = encoding.pe
        encoding(torch.zeros((1, 4, 3, 2), dtype=torch.float64))

        self.assertIsNot(encoding.pe, first)
        self.assertEqual(encoding.pe.shape, (1, 4, 3, 2))
        self.assertEqual(encoding.pe.dtype, torch.float64)


class VocabularyAndPredictionTests(unittest.TestCase):
    def test_i2w_keys_are_normalized_once(self):
        normalize = getattr(modeling_smt, "_normalize_i2w")

        self.assertEqual(normalize({"0": "<bos>", 1: "note"}), {0: "<bos>", 1: "note"})

    def test_i2w_conflicting_numeric_keys_are_rejected(self):
        normalize = getattr(modeling_smt, "_normalize_i2w")

        with self.assertRaisesRegex(ValueError, "Conflicting i2w entries"):
            normalize({1: "note", "1": "rest"})

    def test_predict_accepts_legacy_convert_to_str_with_integer_vocabulary(self):
        model = SimpleNamespace(
            w2i={"<bos>": 0},
            i2w={0: "<bos>", 1: "note", 2: "<eos>"},
            maxlen=4,
        )
        model.forward_encoder = lambda input: torch.zeros((input.shape[0], 2, 4), device=input.device)

        def forward_decoder(_encoder_output, predicted_sequence, **_kwargs):
            token_id = 1 if predicted_sequence.shape[1] == 1 else 2
            logits = torch.full((1, 3, 1), -10.0)
            logits[:, token_id, :] = 10.0
            return SimpleNamespace(logits=logits)

        model.forward_decoder = forward_decoder

        with self.assertWarnsRegex(DeprecationWarning, "convert_to_str"):
            sequence, _ = SMTFoundationModelForCausalLM.predict(
                model,
                torch.zeros((1, 3, 2, 2)),
                convert_to_str=True,
            )

        self.assertEqual(sequence, ["note"])


class EncoderTrainabilityTests(unittest.TestCase):
    def test_unfreeze_encoder_changes_requires_grad(self):
        model = SimpleNamespace(encoder=torch.nn.Linear(2, 2))

        SMTFoundationModelForCausalLM.freeze_encoder(model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        SMTFoundationModelForCausalLM.unfreeze_encoder(model)

        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
