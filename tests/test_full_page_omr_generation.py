import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from experiments.full_page_omr.eval.eval_functions import CanonicalTokenStream
from experiments.full_page_omr.smt_foundation.modeling_smt import (
    GenerationResult,
    SMTFoundationModelForCausalLM,
)
from experiments.full_page_omr.smt_trainer import SMTPP_Trainer


def _uncached_stub(always_token=None, maxlen=4):
    model = SimpleNamespace(
        w2i={"<bos>": 0},
        i2w={0: "<bos>", 1: "note", 2: "<eos>"},
        maxlen=maxlen,
    )
    model.forward_encoder = lambda input: torch.zeros(
        (input.shape[0], 2, 4), device=input.device
    )

    def forward_decoder(_encoder_output, predicted_sequence, **_kwargs):
        token_id = always_token
        if token_id is None:
            token_id = 1 if predicted_sequence.shape[1] == 1 else 2
        logits = torch.full((1, 3, 1), -10.0)
        logits[:, token_id, :] = 10.0
        return SimpleNamespace(logits=logits)

    model.forward_decoder = forward_decoder
    return model


class RawGenerationTests(unittest.TestCase):
    def test_uncached_generation_returns_bos_content_and_eos_ids(self):
        model = _uncached_stub()

        result = SMTFoundationModelForCausalLM.generate_token_ids(
            model,
            torch.zeros((1, 3, 2, 2)),
            use_incremental=False,
        )

        self.assertEqual(result.token_ids, (0, 1, 2))
        self.assertTrue(result.terminated_by_eos)
        self.assertFalse(result.truncated)
        self.assertEqual(result.output.logits.shape, (1, 3, 1))

    def test_uncached_generation_marks_missing_eos_as_truncated(self):
        model = _uncached_stub(always_token=1, maxlen=3)

        result = SMTFoundationModelForCausalLM.generate_token_ids(
            model,
            torch.zeros((1, 3, 2, 2)),
            use_incremental=False,
        )

        self.assertEqual(result.token_ids, (0, 1, 1))
        self.assertFalse(result.terminated_by_eos)
        self.assertTrue(result.truncated)

    def test_predict_remains_a_string_compatibility_adapter(self):
        model = _uncached_stub()

        sequence, output = SMTFoundationModelForCausalLM.predict(
            model,
            torch.zeros((1, 3, 2, 2)),
        )

        self.assertEqual(sequence, ["note"])
        self.assertEqual(output.logits.shape, (1, 3, 1))


class _MetricModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.i2w = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "note"}
        self.maxlen = 4

    def freeze_encoder(self):
        return None

    def unfreeze_encoder(self):
        return None

    def generate_token_ids(self, input):
        del input
        return GenerationResult(
            token_ids=(1, 3, 2),
            output=SimpleNamespace(logits=torch.zeros((1, 4, 1))),
            terminated_by_eos=True,
            truncated=False,
        )

    def forward(self, x, decoder_input, labels=None):
        del x, decoder_input, labels
        return SimpleNamespace(loss=self.weight)


class TrainerMetricIntegrationTests(unittest.TestCase):
    def test_validation_stores_canonical_streams_and_logs_only_v2_metrics(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _MetricModel(),
            encoder_training_mode="linear_probe",
        )
        module.log = Mock()
        batch = (
            torch.zeros((1, 3, 2, 2)),
            torch.tensor([[1, 3, 2]]),
            torch.tensor([[3, 2, 0]]),
        )

        module.validation_step(batch)

        expected = CanonicalTokenStream(("note",), True, False)
        self.assertEqual(module.preds, [expected])
        self.assertEqual(module.grtrs, [expected])

        ser = module.on_validation_epoch_end()

        self.assertEqual(ser, 0.0)
        self.assertEqual(
            [call.args[0] for call in module.log.call_args_list],
            ["val_CER_v2", "val_SER_v2", "val_LER_v2"],
        )
        self.assertTrue(all(call.args[1] == 0.0 for call in module.log.call_args_list))
        self.assertEqual(module.preds, [])
        self.assertEqual(module.grtrs, [])


if __name__ == "__main__":
    unittest.main()
