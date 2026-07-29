import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from lightning.pytorch.trainer.states import TrainerFn

from experiments.full_page_omr import data, entrypoint, finetune
from experiments.full_page_omr import smt_trainer
from experiments.full_page_omr.optimization import optimizer_protocol_metadata
from experiments.full_page_omr.pdmx_data import (
    PDMXConsumptionAuditCallback,
    PDMXPretrainingDataModule,
    PDMXVirtualEpochCallback,
)
from experiments.full_page_omr.smt_trainer import SMTPP_Trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_snapshot(model, **overrides):
    metadata = finetune._build_protocol_metadata(
        max_steps=4_000_000,
        validation_every_n_epochs=2_000,
        from_checkpoint=None,
        starting_weights=None,
        encoder_training_mode="fine_tune",
        encoder_unfreeze_step=120_000,
        resolution=1024,
        reduce_ratio=0.5,
        batch_size=1,
        expected_training_batches_per_epoch=83,
        curriculum_steady_mixture_step=320_000,
        finetuning_technique="CL",
        attention_backend="auto",
        tokenization_mode="bekern",
        num_workers=24,
        checkpoint_every_n_epochs=100,
        optimizer_metadata=optimizer_protocol_metadata(
            model,
            finetune.AdamWWSDConfig(),
        ),
    )
    snapshot = metadata["protocol_snapshot"]
    for path, value in overrides.items():
        target = snapshot
        parts = path.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return snapshot


def _full_resume_payload(model, *, global_step=40, samples_seen=None,
                         curriculum_step_offset=0, snapshot=None):
    if snapshot is None:
        snapshot = _protocol_snapshot(model)
    module = SMTPP_Trainer(
        SimpleNamespace(padding_token=0),
        model,
        encoder_unfreeze_step=120_000,
        curriculum_step_offset=curriculum_step_offset,
        protocol_snapshot=snapshot,
    )
    configured = module.configure_optimizers()
    optimizer_state = configured["optimizer"].state_dict()
    scheduler = configured["lr_scheduler"]["scheduler"]
    multiplier = scheduler.lr_lambdas[0](global_step)
    current_lrs = [base_lr * multiplier for base_lr in scheduler.base_lrs]
    for group, current_lr in zip(optimizer_state["param_groups"], current_lrs):
        group["lr"] = current_lr
    scheduler_state = scheduler.state_dict()
    scheduler_state["last_epoch"] = global_step
    scheduler_state["_step_count"] = global_step + 1
    scheduler_state["_last_lr"] = current_lrs
    return {
        "global_step": global_step,
        "full_page_omr_samples_seen": (
            global_step if samples_seen is None else samples_seen
        ),
        "hyper_parameters": dict(module.hparams),
        "optimizer_states": [optimizer_state],
        "lr_schedulers": [scheduler_state],
    }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.task_bias = torch.nn.Parameter(torch.zeros(()))
        self.encoder = torch.nn.Linear(1, 1)

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward(self, x, decoder_input, labels=None):
        return SimpleNamespace(loss=self.weight * 0.5)


class FullPageOMRCheckpointTests(unittest.TestCase):
    def test_pdmx_regime_is_registered(self):
        self.assertIs(
            finetune.DATASETS_TYPE["PDMX"],
            PDMXPretrainingDataModule,
        )

    def test_final_evaluation_skips_explicit_no_test_datamodule(self):
        trainer = Mock()
        data_module = SimpleNamespace(has_test_split=False)

        ran = finetune._run_test_if_available(
            trainer,
            Mock(),
            data_module,
            "model.ckpt",
        )

        self.assertFalse(ran)
        trainer.test.assert_not_called()

    def test_final_evaluation_preserves_legacy_default(self):
        trainer = Mock()

        ran = finetune._run_test_if_available(
            trainer,
            Mock(),
            object(),
            "model.ckpt",
        )

        self.assertTrue(ran)
        trainer.test.assert_called_once()

    def test_data_protocol_metadata_is_forwarded_without_aliasing(self):
        source = {
            "dataset_revision": "fixed",
            "vocab_sha256": "a" * 64,
            "stream_resume_mode": "virtual_epoch_boundary",
            "steps_per_epoch": 10_000,
        }
        data_module = SimpleNamespace(protocol_metadata=lambda: source)

        metadata = finetune._data_protocol_metadata(data_module)
        metadata["dataset_revision"] = "mutated"

        self.assertEqual(source["dataset_revision"], "fixed")

    def test_pdmx_protocol_is_embedded_in_checkpoint_snapshot(self):
        data_protocol = {
            "dataset_revision": "7" * 40,
            "dataset_manifest_sha256": "a" * 64,
            "vocab_size": 223,
            "vocab_sha256": "b" * 64,
            "stream_resume_mode": "virtual_epoch_boundary",
            "steps_per_epoch": 10_000,
        }

        metadata = finetune._build_protocol_metadata(
            max_steps=4_000_000,
            validation_every_n_epochs=2_000,
            from_checkpoint=None,
            starting_weights=None,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=0,
            resolution=1024,
            reduce_ratio=1.0,
            batch_size=1,
            expected_training_batches_per_epoch=10_000,
            curriculum_steady_mixture_step=320_000,
            finetuning_technique="PDMX",
            attention_backend="auto",
            tokenization_mode="bekern",
            num_workers=2,
            checkpoint_every_n_epochs=100,
            optimizer_metadata=optimizer_protocol_metadata(
                _TinyModel(),
                finetune.AdamWWSDConfig(),
            ),
            data_protocol=data_protocol,
        )

        self.assertEqual(metadata["vocab_size"], 223)
        self.assertEqual(
            metadata["protocol_snapshot"]["data"]["vocab_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            metadata["protocol_snapshot"]["data"]["stream_resume_mode"],
            "virtual_epoch_boundary",
        )

    def test_pdmx_full_resume_requires_virtual_epoch_boundary(self):
        with self.assertRaisesRegex(ValueError, "virtual epoch boundary"):
            finetune._validate_stream_resume_boundary(
                samples_seen=10_001,
                steps_per_epoch=10_000,
                mode="virtual_epoch_boundary",
            )

        finetune._validate_stream_resume_boundary(
            samples_seen=20_000,
            steps_per_epoch=10_000,
            mode="virtual_epoch_boundary",
        )

    def test_pdmx_callbacks_are_capability_based(self):
        data_module = SimpleNamespace(
            set_train_epoch=Mock(),
            protocol_metadata=lambda: {
                "stream_resume_mode": "virtual_epoch_boundary"
            },
        )

        callbacks = finetune._data_callbacks(
            data_module,
            Path("run-instance"),
        )

        self.assertEqual(len(callbacks), 2)
        self.assertIsInstance(callbacks[0], PDMXVirtualEpochCallback)
        self.assertIsInstance(callbacks[1], PDMXConsumptionAuditCallback)

    def test_validation_step_accepts_optional_batch_metadata(self):
        model = _TinyModel()
        model.i2w = {0: "<bos>", 1: "4c", 2: "<eos>"}
        model.maxlen = 8
        model.generate_token_ids = Mock(
            return_value=SimpleNamespace(token_ids=[0, 1, 2])
        )
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="linear_probe",
        )
        batch = (
            torch.zeros(1, 3, 2, 2),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[0, 1, 2]]),
            {"renderer": "verovio"},
        )

        module.validation_step(batch)

        self.assertEqual(len(module.grtrs), 1)

    def test_public_entrypoint_forwards_checkpoint_interval_and_attention_backend(self):
        signature = inspect.signature(entrypoint.run)
        self.assertEqual(signature.parameters["checkpoint_every_n_epochs"].default, 100)
        self.assertEqual(signature.parameters["attention_backend"].default, "auto")
        self.assertEqual(signature.parameters["encoder_training_mode"].default, "fine_tune")
        self.assertEqual(signature.parameters["validation_every_n_epochs"].default, 2_000)
        self.assertEqual(signature.parameters["max_steps"].default, 4_000_000)
        self.assertEqual(signature.parameters["task_learning_rate"].default, None)
        self.assertEqual(signature.parameters["encoder_learning_rate"].default, 1e-5)
        self.assertEqual(signature.parameters["weight_decay"].default, 0.01)
        self.assertEqual(signature.parameters["wsd_warmup_steps"].default, 10_000)
        self.assertEqual(signature.parameters["wsd_decay_steps"].default, 400_000)
        self.assertEqual(
            signature.parameters["protocol_version"].default,
            "full_page_omr_adamw_wsd_4m_v1",
        )

        config_path = REPO_ROOT / "experiments/full_page_omr/config/Polish_Scores/finetuning.json"
        with patch.object(entrypoint, "_launch") as launch:
            entrypoint.run(
                str(config_path),
                "test-run",
                checkpoint_every_n_epochs=37,
                attention_backend="sdpa",
                encoder_training_mode="linear_probe",
                validation_every_n_epochs=2500,
                task_learning_rate=2e-4,
                encoder_learning_rate=2e-5,
                weight_decay=0.02,
                wsd_warmup_steps=20_000,
                wsd_decay_steps=500_000,
                protocol_version="full_page_omr_resize_v1",
                starting_weights="weights.ckpt",
                source_curriculum_step=282200,
                source_checkpoint_sha256="a" * 64,
                source_vocab_manifest="source-vocab.json",
            )

        self.assertEqual(launch.call_args.kwargs["checkpoint_every_n_epochs"], 37)
        self.assertEqual(launch.call_args.kwargs["attention_backend"], "sdpa")
        self.assertEqual(launch.call_args.kwargs["encoder_training_mode"], "linear_probe")
        self.assertEqual(launch.call_args.kwargs["validation_every_n_epochs"], 2500)
        self.assertEqual(launch.call_args.kwargs["task_learning_rate"], 2e-4)
        self.assertEqual(launch.call_args.kwargs["encoder_learning_rate"], 2e-5)
        self.assertEqual(launch.call_args.kwargs["weight_decay"], 0.02)
        self.assertEqual(launch.call_args.kwargs["wsd_warmup_steps"], 20_000)
        self.assertEqual(launch.call_args.kwargs["wsd_decay_steps"], 500_000)
        self.assertEqual(
            launch.call_args.kwargs["protocol_version"],
            "full_page_omr_resize_v1",
        )
        self.assertEqual(launch.call_args.kwargs["source_curriculum_step"], 282200)
        self.assertEqual(launch.call_args.kwargs["source_checkpoint_sha256"], "a" * 64)
        self.assertEqual(
            launch.call_args.kwargs["source_vocab_manifest"],
            "source-vocab.json",
        )

    def test_launch_forwards_checkpoint_interval_to_main(self):
        config = {
            "data": {
                "data_path": "example/dataset",
                "batch_size": 1,
                "vocab_name": "Example",
                "num_workers": 0,
                "tokenization_mode": "bekern",
                "reduce_ratio": 0.5,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="ascii")
            with patch.object(finetune, "main") as main:
                finetune.launch(
                    str(config_path),
                    "test-run",
                    checkpoint_every_n_epochs=37,
                    encoder_training_mode="linear_probe",
                    validation_every_n_epochs=2500,
                    task_learning_rate=2e-4,
                    encoder_learning_rate=2e-5,
                    weight_decay=0.02,
                    wsd_warmup_steps=20_000,
                    wsd_decay_steps=500_000,
                    protocol_version="full_page_omr_resize_v1",
                    starting_weights="weights.ckpt",
                    source_curriculum_step=282200,
                    source_checkpoint_sha256="a" * 64,
                    source_vocab_manifest="source-vocab.json",
                )

        self.assertEqual(main.call_args.kwargs["checkpoint_every_n_epochs"], 37)
        self.assertEqual(main.call_args.kwargs["encoder_training_mode"], "linear_probe")
        self.assertEqual(main.call_args.kwargs["validation_every_n_epochs"], 2500)
        self.assertEqual(main.call_args.kwargs["max_steps"], 4_000_000)
        self.assertEqual(main.call_args.kwargs["task_learning_rate"], 2e-4)
        self.assertEqual(main.call_args.kwargs["encoder_learning_rate"], 2e-5)
        self.assertEqual(main.call_args.kwargs["weight_decay"], 0.02)
        self.assertEqual(main.call_args.kwargs["wsd_warmup_steps"], 20_000)
        self.assertEqual(main.call_args.kwargs["wsd_decay_steps"], 500_000)
        self.assertEqual(
            main.call_args.kwargs["protocol_version"],
            "full_page_omr_resize_v1",
        )
        self.assertEqual(main.call_args.kwargs["source_curriculum_step"], 282200)
        self.assertEqual(main.call_args.kwargs["source_checkpoint_sha256"], "a" * 64)
        self.assertEqual(
            main.call_args.kwargs["source_vocab_manifest"],
            "source-vocab.json",
        )

    def test_source_vocab_manifest_requires_starting_weights(self):
        with self.assertRaisesRegex(ValueError, "starting_weights"):
            finetune._validate_source_vocab_manifest(
                None,
                "source-vocab.json",
            )
        self.assertEqual(
            finetune._validate_source_vocab_manifest(
                "weights.ckpt",
                "source-vocab.json",
            ),
            "source-vocab.json",
        )

    def test_main_uses_fresh_wrapper_for_vocabulary_migration(self):
        class SizedDataset(SimpleNamespace):
            def __len__(self):
                return 83

        data_module = SimpleNamespace(
            train_dataset=SizedDataset(
                w2i={"<pad>": 0},
                i2w={0: "<pad>"},
            ),
            encoder_unfreeze_step=0,
            curriculum_step_offset=0,
            tokenization_mode="bekern",
            batch_size=1,
            num_workers=0,
            vocab_manifest_path=Path("target-vocab.json"),
        )
        config = SimpleNamespace(
            data=SimpleNamespace(skip_steps=0, reduce_ratio=1.0)
        )
        model = _TinyModel()
        model.encoder.config = SimpleNamespace(patch_size=16)
        wrapper = Mock()
        trainer_class = Mock(return_value=wrapper)

        with (
            patch.object(finetune, "_validate_run_contract", return_value=None),
            patch.dict(finetune.DATASETS_TYPE, {"CL": lambda _: data_module}),
            patch.object(finetune, "set_up_processor"),
            patch.object(finetune, "SMTFoundationConfig", return_value=object()),
            patch.object(
                finetune,
                "SMTFoundationModelForCausalLM",
                return_value=model,
            ),
            patch.object(finetune, "SMTPP_Trainer", trainer_class),
            patch.object(
                finetune,
                "load_vocabulary_aware_weights",
                side_effect=RuntimeError("migration captured"),
            ) as migrate,
        ):
            with self.assertRaisesRegex(RuntimeError, "migration captured"):
                finetune.main(
                    config,
                    "migration-run",
                    starting_weights="weights.ckpt",
                    source_vocab_manifest="source-vocab.json",
                    protocol_version="vocab-migration-v1",
                )

        trainer_class.assert_called_once()
        trainer_class.load_from_checkpoint.assert_not_called()
        self.assertEqual(
            migrate.call_args.kwargs["source_vocab_manifest"],
            "source-vocab.json",
        )
        self.assertEqual(
            migrate.call_args.kwargs["target_vocab_manifest"],
            Path("target-vocab.json"),
        )

    def test_checkpoint_interval_must_be_a_positive_integer(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finetune._validate_checkpoint_every_n_epochs(value)

        self.assertEqual(finetune._validate_checkpoint_every_n_epochs(37), 37)

    def test_validation_interval_and_production_max_steps_are_validated(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                finetune._validate_validation_every_n_epochs(value)
        self.assertEqual(finetune._validate_validation_every_n_epochs(2000), 2000)

        for value in (0, -1, True, 1.5):
            with self.subTest(max_steps=value), self.assertRaises(ValueError):
                finetune._validate_max_steps(value, train=True)
        self.assertEqual(finetune._validate_max_steps(4_000_000, train=True), 4_000_000)
        self.assertEqual(finetune._validate_max_steps(-1, train=False), -1)

    def test_deprecated_learning_rate_alias_is_normalized(self):
        self.assertEqual(finetune._normalize_task_learning_rate(None, None), 1e-4)
        self.assertEqual(finetune._normalize_task_learning_rate(2e-4, None), 2e-4)
        self.assertEqual(finetune._normalize_task_learning_rate(None, 3e-4), 3e-4)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            finetune._normalize_task_learning_rate(2e-4, 3e-4)
        for value in (0, -1e-4, True, "1e-4"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive"):
                finetune._normalize_task_learning_rate(value, None)

    def test_canonical_protocol_still_rejects_optimizer_overrides(self):
        locked = finetune.AdamWWSDConfig()
        mutations = {
            "task_learning_rate": 2e-4,
            "encoder_learning_rate": 2e-5,
            "weight_decay": 0.02,
            "betas": (0.8, 0.99),
            "eps": 1e-7,
            "warmup_steps": 10_001,
            "warmup_type": "cosine",
            "decay_type": "linear",
        }
        for field_name, value in mutations.items():
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "new protocol_version"):
                    finetune._validate_canonical_protocol_contract(
                        finetune.PROTOCOL_VERSION,
                        replace(locked, **{field_name: value}),
                        validation_every_n_epochs=2_000,
                    )

    def test_canonical_protocol_allows_validation_cadence_override(self):
        finetune._validate_canonical_protocol_contract(
            finetune.PROTOCOL_VERSION,
            finetune.AdamWWSDConfig(),
            validation_every_n_epochs=200,
        )

    def test_noncanonical_protocol_allows_optimizer_and_cadence_overrides(self):
        finetune._validate_canonical_protocol_contract(
            "full_page_omr_experiment_v1",
            replace(
                finetune.AdamWWSDConfig(),
                task_learning_rate=2e-4,
                max_steps=4_000_001,
            ),
            validation_every_n_epochs=2_001,
        )

    def test_pdmx_stream_requires_validation_each_virtual_epoch(self):
        pdmx_data = SimpleNamespace(
            stream_resume_mode="virtual_epoch_boundary"
        )

        with self.assertRaisesRegex(
            ValueError,
            "validation_every_n_epochs=1",
        ):
            finetune._validate_data_regime_contract(
                pdmx_data,
                protocol_version="full_page_omr_pdmx_v1",
                validation_every_n_epochs=2_000,
            )

        with self.assertRaisesRegex(ValueError, "distinct protocol_version"):
            finetune._validate_data_regime_contract(
                pdmx_data,
                protocol_version=finetune.PROTOCOL_VERSION,
                validation_every_n_epochs=1,
            )

        finetune._validate_data_regime_contract(
            pdmx_data,
            protocol_version="full_page_omr_pdmx_v1",
            validation_every_n_epochs=1,
        )
        finetune._validate_data_regime_contract(
            SimpleNamespace(),
            protocol_version=finetune.PROTOCOL_VERSION,
            validation_every_n_epochs=2_000,
        )

    def test_canonical_protocol_allows_wsd_endpoint_and_decay_overrides(self):
        finetune._validate_canonical_protocol_contract(
            finetune.PROTOCOL_VERSION,
            replace(
                finetune.AdamWWSDConfig(),
                max_steps=2_000_000,
                decay_steps=340_000,
                min_lr_ratio=0.1,
            ),
            validation_every_n_epochs=2_000,
        )

    def test_main_passes_complete_protocol_snapshot_to_full_resume_contract(self):
        class SizedDataset(SimpleNamespace):
            def __len__(self):
                return 83

        data_module = SimpleNamespace(
            train_dataset=SizedDataset(w2i={"<pad>": 0}, i2w={0: "<pad>"}),
            encoder_unfreeze_step=120_000,
            curriculum_step_offset=0,
            tokenization_mode="bekern",
            batch_size=1,
            num_workers=0,
        )
        config = SimpleNamespace(data=SimpleNamespace(skip_steps=0, reduce_ratio=0.5))
        model = _TinyModel()
        model.encoder.config = SimpleNamespace(patch_size=16)
        protocol_version = "full_page_omr_optimizer_override_v1"
        with (
            patch.dict(finetune.DATASETS_TYPE, {"CL": lambda _: data_module}),
            patch.object(finetune, "set_up_processor"),
            patch.object(finetune, "SMTFoundationConfig", return_value=object()),
            patch.object(
                finetune,
                "SMTFoundationModelForCausalLM",
                return_value=model,
            ),
            patch.object(
                finetune,
                "_validate_run_contract",
                side_effect=RuntimeError("contract captured"),
            ) as validate_contract,
        ):
            with self.assertRaisesRegex(RuntimeError, "contract captured"):
                finetune.main(
                    config,
                    "test-run",
                    from_checkpoint="resume.ckpt",
                    max_steps=4_000_000,
                    task_learning_rate=2e-4,
                    encoder_learning_rate=2e-5,
                    weight_decay=0.02,
                    wsd_warmup_steps=20_000,
                    wsd_decay_steps=500_000,
                    protocol_version=protocol_version,
                )

        snapshot = validate_contract.call_args.kwargs["expected_protocol_snapshot"]
        self.assertEqual(snapshot["protocol_version"], protocol_version)
        self.assertEqual(snapshot["optimizer"]["implementation"], "torch.optim.AdamW")
        self.assertEqual(snapshot["optimizer"]["betas"], [0.9, 0.999])
        self.assertEqual(snapshot["optimizer"]["eps"], 1e-8)
        self.assertFalse(snapshot["optimizer"]["amsgrad"])
        self.assertEqual(snapshot["scheduler"]["num_cycles"], 0.5)
        self.assertEqual(snapshot["scheduler"]["warmup_steps"], 20_000)

    def test_main_passes_optimizer_arguments_to_fresh_and_weights_only_wrappers(self):
        class SizedDataset(SimpleNamespace):
            def __len__(self):
                return 83

        train_dataset = SizedDataset(w2i={"<pad>": 0}, i2w={0: "<pad>"})
        data_module = SimpleNamespace(
            train_dataset=train_dataset,
            encoder_unfreeze_step=120_000,
            curriculum_step_offset=0,
            tokenization_mode="bekern",
            batch_size=1,
            num_workers=0,
        )
        config = SimpleNamespace(data=SimpleNamespace(skip_steps=0, reduce_ratio=0.5))
        model = _TinyModel()
        model.encoder.config = SimpleNamespace(patch_size=16)
        protocol_version = "full_page_omr_optimizer_override_v1"
        optimizer_kwargs = {
            "run_protocol_version": protocol_version,
            "task_learning_rate": 2e-4,
            "encoder_learning_rate": 2e-5,
            "weight_decay": 0.02,
            "max_steps": 4_000_000,
            "wsd_warmup_steps": 20_000,
            "wsd_decay_steps": 500_000,
            "wsd_warmup_type": "linear",
            "wsd_decay_type": "cosine",
            "wsd_min_lr_ratio": 0.0,
        }

        for starting_weights in (None, "weights.ckpt"):
            with self.subTest(starting_weights=starting_weights):
                trainer_class = Mock()
                trainer_class.side_effect = RuntimeError("wrapper captured")
                trainer_class.load_from_checkpoint.side_effect = RuntimeError(
                    "wrapper captured"
                )
                with (
                    patch.object(finetune, "_validate_run_contract", return_value=None),
                    patch.dict(finetune.DATASETS_TYPE, {"CL": lambda _: data_module}),
                    patch.object(finetune, "set_up_processor"),
                    patch.object(finetune, "SMTFoundationConfig", return_value=object()),
                    patch.object(
                        finetune,
                        "SMTFoundationModelForCausalLM",
                        return_value=model,
                    ),
                    patch.object(finetune, "SMTPP_Trainer", trainer_class),
                ):
                    with self.assertRaisesRegex(RuntimeError, "wrapper captured"):
                        finetune.main(
                            config,
                            "test-run",
                            starting_weights=starting_weights,
                            task_learning_rate=2e-4,
                            encoder_learning_rate=2e-5,
                            weight_decay=0.02,
                            wsd_warmup_steps=20_000,
                            wsd_decay_steps=500_000,
                            protocol_version=protocol_version,
                        )

                call = (
                    trainer_class.call_args
                    if starting_weights is None
                    else trainer_class.load_from_checkpoint.call_args
                )
                for key, value in optimizer_kwargs.items():
                    self.assertEqual(call.kwargs[key], value)

    def test_eval_only_minus_one_uses_canonical_wsd_total_and_reaches_test_setup(self):
        class SizedDataset(SimpleNamespace):
            def __len__(self):
                return 83

        train_dataset = SizedDataset(w2i={"<pad>": 0}, i2w={0: "<pad>"})
        data_module = SimpleNamespace(
            train_dataset=train_dataset,
            encoder_unfreeze_step=120_000,
            curriculum_step_offset=0,
            tokenization_mode="bekern",
            batch_size=1,
            num_workers=0,
        )
        config = SimpleNamespace(data=SimpleNamespace(skip_steps=0, reduce_ratio=0.5))
        model = _TinyModel()
        model.encoder.config = SimpleNamespace(patch_size=16)
        wrapper = Mock()
        wrapper.optimizer_protocol_metadata.return_value = {
            "optimizer": "AdamW",
            "optimizer_implementation": "torch.optim.AdamW",
            "torch_version": str(torch.__version__),
            "wsd_max_steps": 4_000_000,
        }
        trainer = Mock()
        protocol_path = Path("logs/protocol.json")

        with (
            patch.dict(finetune.DATASETS_TYPE, {"CL": lambda _: data_module}),
            patch.object(finetune, "set_up_processor"),
            patch.object(finetune, "SMTFoundationConfig", return_value=object()),
            patch.object(
                finetune,
                "SMTFoundationModelForCausalLM",
                return_value=model,
            ),
            patch.object(finetune, "SMTPP_Trainer", return_value=wrapper) as module,
            patch.object(finetune, "_build_epoch_checkpointer", return_value=object()),
            patch.object(
                finetune,
                "_build_metric_checkpointer",
                return_value=SimpleNamespace(best_model_path=""),
            ),
            patch.object(finetune, "_write_resize_audit", return_value=None),
            patch.object(
                finetune,
                "_write_protocol_metadata",
                return_value=protocol_path,
            ) as write_protocol,
            patch.object(finetune, "WandbLogger") as wandb_logger,
            patch.object(finetune, "Trainer", return_value=trainer) as trainer_class,
            patch.object(finetune, "_run_test") as run_test,
        ):
            finetune.main(config, "eval-run", train=False, max_steps=-1)

        self.assertEqual(module.call_args.kwargs["max_steps"], 4_000_000)
        self.assertEqual(trainer_class.call_args.kwargs["max_steps"], -1)
        metadata = write_protocol.call_args.args[1]
        self.assertEqual(metadata["max_steps"], 4_000_000)
        self.assertEqual(metadata["trainer_max_steps"], -1)
        self.assertIs(
            wandb_logger.return_value.log_hyperparams.call_args.args[0],
            metadata,
        )
        self.assertEqual(metadata["optimizer_implementation"], "torch.optim.AdamW")
        self.assertEqual(metadata["torch_version"], str(torch.__version__))
        self.assertEqual(
            module.call_args.kwargs["protocol_snapshot"],
            metadata["protocol_snapshot"],
        )
        run_test.assert_called_once_with(trainer, wrapper, data_module, None)

    def test_eval_only_from_legacy_checkpoint_is_weights_loading_with_provenance(self):
        class SizedDataset(SimpleNamespace):
            def __len__(self):
                return 83

        train_dataset = SizedDataset(w2i={"<pad>": 0}, i2w={0: "<pad>"})
        data_module = SimpleNamespace(
            train_dataset=train_dataset,
            encoder_unfreeze_step=120_000,
            curriculum_step_offset=0,
            tokenization_mode="bekern",
            batch_size=1,
            num_workers=0,
        )
        config = SimpleNamespace(data=SimpleNamespace(skip_steps=0, reduce_ratio=0.5))
        model = _TinyModel()
        model.encoder.config = SimpleNamespace(patch_size=1)
        wrapper = Mock()
        trainer = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "legacy-weights.ckpt"
            torch.save({"state_dict": {"model.weight": torch.ones(())}}, checkpoint)
            with (
                patch.dict(finetune.DATASETS_TYPE, {"CL": lambda _: data_module}),
                patch.object(finetune, "set_up_processor"),
                patch.object(finetune, "SMTFoundationConfig", return_value=object()),
                patch.object(
                    finetune,
                    "SMTFoundationModelForCausalLM",
                    return_value=model,
                ),
                patch.object(finetune, "SMTPP_Trainer", return_value=wrapper),
                patch.object(finetune, "_build_epoch_checkpointer", return_value=object()),
                patch.object(
                    finetune,
                    "_build_metric_checkpointer",
                    return_value=SimpleNamespace(best_model_path=""),
                ),
                patch.object(finetune, "_write_resize_audit", return_value=None),
                patch.object(
                    finetune,
                    "_write_protocol_metadata",
                    return_value=Path("logs/protocol.json"),
                ) as write_protocol,
                patch.object(finetune, "WandbLogger"),
                patch.object(finetune, "Trainer", return_value=trainer),
                patch.object(finetune, "_run_test") as run_test,
            ):
                finetune.main(
                    config,
                    "legacy-eval",
                    train=False,
                    max_steps=-1,
                    from_checkpoint=str(checkpoint),
                )

            metadata = write_protocol.call_args.args[1]
            self.assertEqual(
                metadata["checkpoint_load_mode"],
                "evaluation_weights_only",
            )
            self.assertEqual(metadata["checkpoint_sha256"], _sha256_file(checkpoint))
            self.assertIsNone(metadata["checkpoint_global_step"])
            run_test.assert_called_once_with(
                trainer,
                wrapper,
                data_module,
                str(checkpoint),
            )

    def test_encoder_training_mode_is_validated(self):
        self.assertEqual(finetune._validate_encoder_training_mode("fine_tune"), "fine_tune")
        self.assertEqual(finetune._validate_encoder_training_mode("linear_probe"), "linear_probe")
        for value in ("finetune", "CL", "", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finetune._validate_encoder_training_mode(value)

    def test_full_resume_and_starting_weights_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            finetune.main(
                config=object(),
                experiment_name="test-run",
                from_checkpoint="resume.ckpt",
                starting_weights="initial.ckpt",
            )

    def test_full_resume_endpoint_must_exceed_checkpoint_step(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save(
                _full_resume_payload(
                    model,
                    global_step=4_000_000,
                    snapshot=snapshot,
                ),
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "max_steps.*global_step"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                    from_checkpoint=str(checkpoint),
                    starting_weights=None,
                    max_steps=4_000_000,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                    expected_protocol_snapshot=snapshot,
                    expected_model=model,
                    optimizer_config=finetune.AdamWWSDConfig(),
                )

    def test_full_resume_rejects_checkpoint_without_protocol_snapshot(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        payload = _full_resume_payload(model, snapshot=snapshot)
        del payload["hyper_parameters"]["protocol_snapshot"]
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "legacy-adam.ckpt"
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "protocol_snapshot"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                    from_checkpoint=str(checkpoint),
                    starting_weights=None,
                    max_steps=4_000_000,
                    train=True,
                    protocol_version="full_page_omr_adamw_wsd_4m_v1",
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                    expected_protocol_snapshot=snapshot,
                    expected_model=model,
                    optimizer_config=finetune.AdamWWSDConfig(),
                )

    def test_full_resume_requires_global_step_and_samples_seen(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        for missing_key in ("global_step", "full_page_omr_samples_seen"):
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory() as tmpdir:
                payload = _full_resume_payload(model, snapshot=snapshot)
                del payload[missing_key]
                checkpoint = Path(tmpdir) / "incomplete.ckpt"
                torch.save(payload, checkpoint)
                with self.assertRaisesRegex(ValueError, "global_step|samples_seen"):
                    finetune._validate_run_contract(
                        config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                        from_checkpoint=str(checkpoint),
                        starting_weights=None,
                        max_steps=4_000_000,
                        train=True,
                        protocol_version=finetune.PROTOCOL_VERSION,
                        source_curriculum_step=None,
                        expected_protocol_snapshot=snapshot,
                        expected_model=model,
                        optimizer_config=finetune.AdamWWSDConfig(),
                    )

    def test_run_contract_rejects_corrupt_optimizer_state_before_trainer(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        payload = _full_resume_payload(model, snapshot=snapshot)
        payload["optimizer_states"][0]["param_groups"][0]["lr"] = 123.0
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "corrupt-state.ckpt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "parameter-group mismatch"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                    from_checkpoint=str(checkpoint),
                    starting_weights=None,
                    max_steps=4_000_000,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    expected_protocol_snapshot=snapshot,
                    expected_model=model,
                    optimizer_config=finetune.AdamWWSDConfig(),
                )

    def test_full_resume_accepts_retargeted_wsd_endpoint_and_decay(self):
        model = _TinyModel()
        saved_snapshot = _protocol_snapshot(model)
        expected_snapshot = _protocol_snapshot(
            model,
            scheduler__max_steps=2_000_000,
            scheduler__stable_steps=1_650_000,
            scheduler__decay_steps=340_000,
            scheduler__min_lr_ratio=0.1,
            trainer__max_steps=2_000_000,
            validation__expected_count=12,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save(_full_resume_payload(model, snapshot=saved_snapshot), checkpoint)

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                from_checkpoint=str(checkpoint),
                starting_weights=None,
                max_steps=2_000_000,
                train=True,
                protocol_version="full_page_omr_adamw_wsd_4m_v1",
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
                expected_protocol_snapshot=expected_snapshot,
                expected_model=model,
                optimizer_config=replace(
                    finetune.AdamWWSDConfig(),
                    max_steps=2_000_000,
                    decay_steps=340_000,
                    min_lr_ratio=0.1,
                ),
            )

        self.assertEqual(state.global_step, 40)

    def test_full_resume_accepts_runtime_cadence_changes(self):
        model = _TinyModel()
        saved_snapshot = _protocol_snapshot(model)
        expected_snapshot = _protocol_snapshot(
            model,
            validation__every_n_epochs=200,
            validation__first_epoch=200,
            validation__expected_count=240,
            checkpointing__every_n_epochs=25,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save(_full_resume_payload(model, snapshot=saved_snapshot), checkpoint)

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                from_checkpoint=str(checkpoint),
                starting_weights=None,
                max_steps=4_000_000,
                train=True,
                protocol_version=finetune.PROTOCOL_VERSION,
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
                expected_protocol_snapshot=expected_snapshot,
                expected_model=model,
                optimizer_config=finetune.AdamWWSDConfig(),
            )

        self.assertEqual(state.global_step, 40)

    def test_full_resume_requires_complete_expected_protocol_snapshot(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save(_full_resume_payload(model, snapshot=snapshot), checkpoint)
            base = dict(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                from_checkpoint=str(checkpoint),
                starting_weights=None,
                max_steps=4_000_000,
                train=True,
                protocol_version="full_page_omr_adamw_wsd_4m_v1",
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
                expected_model=model,
                optimizer_config=finetune.AdamWWSDConfig(),
            )

            for invalid_snapshot in (None, {}, {"protocol_version": "incomplete"}):
                with self.subTest(snapshot=invalid_snapshot):
                    with self.assertRaisesRegex(ValueError, "protocol_snapshot|snapshot mismatch"):
                        finetune._validate_run_contract(
                            **base,
                            expected_protocol_snapshot=invalid_snapshot,
                        )

    def test_expected_protocol_snapshot_does_not_bypass_protocol_version(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save(_full_resume_payload(model, snapshot=snapshot), checkpoint)

            with self.assertRaisesRegex(ValueError, "protocol snapshot mismatch"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                    from_checkpoint=str(checkpoint),
                    starting_weights=None,
                    max_steps=4_000_000,
                    train=True,
                    protocol_version="different_protocol_v1",
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                    expected_protocol_snapshot=snapshot,
                    expected_model=model,
                    optimizer_config=finetune.AdamWWSDConfig(),
                )

    def test_weights_only_load_ignores_source_optimizer_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "legacy-weights.ckpt"
            torch.save({"global_step": 40}, checkpoint)

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=40)),
                from_checkpoint=None,
                starting_weights=str(checkpoint),
                max_steps=4_000_000,
                train=True,
                protocol_version="weights_only_fork_v1",
                source_curriculum_step=40,
                source_checkpoint_sha256=_sha256_file(checkpoint),
            )

        self.assertIsNone(state.protocol_snapshot)

    def test_vocabulary_migration_starts_target_curriculum_at_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "legacy-weights.ckpt"
            torch.save({"global_step": 40}, checkpoint)

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                from_checkpoint=None,
                starting_weights=str(checkpoint),
                max_steps=4_000_000,
                train=True,
                protocol_version="vocabulary_migration_v1",
                source_curriculum_step=40,
                source_checkpoint_sha256=_sha256_file(checkpoint),
                vocabulary_migration=True,
            )

        self.assertEqual(state.curriculum_step, 40)

    def test_full_resume_rejects_curriculum_offset_mismatch(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model, curriculum__step_offset=10)
        payload = _full_resume_payload(
            model,
            samples_seen=30,
            curriculum_step_offset=10,
            snapshot=snapshot,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            current = Path(tmpdir) / "current.ckpt"
            torch.save(payload, current)
            with self.assertRaisesRegex(ValueError, "curriculum_step_offset"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=9)),
                    from_checkpoint=str(current),
                    starting_weights=None,
                    max_steps=4_000_000,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                    expected_protocol_snapshot=snapshot,
                    expected_model=model,
                    optimizer_config=finetune.AdamWWSDConfig(),
                )

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=10)),
                from_checkpoint=str(current),
                starting_weights=None,
                max_steps=4_000_000,
                train=True,
                protocol_version=finetune.PROTOCOL_VERSION,
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
                expected_protocol_snapshot=snapshot,
                expected_model=model,
                optimizer_config=finetune.AdamWWSDConfig(),
            )

        self.assertEqual(state.curriculum_step_offset, 10)
        self.assertEqual(state.protocol_snapshot, snapshot)

    def test_full_resume_records_checkpoint_provenance(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model, curriculum__step_offset=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            current = Path(tmpdir) / "current.ckpt"
            torch.save(
                _full_resume_payload(
                    model,
                    samples_seen=30,
                    curriculum_step_offset=10,
                    snapshot=snapshot,
                ),
                current,
            )
            current_sha256 = _sha256_file(current)
            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=10)),
                from_checkpoint=str(current),
                starting_weights=None,
                max_steps=4_000_000,
                train=True,
                protocol_version=finetune.PROTOCOL_VERSION,
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
                expected_protocol_snapshot=snapshot,
                expected_model=model,
                optimizer_config=finetune.AdamWWSDConfig(),
            )
            metadata = finetune._build_protocol_metadata(
                max_steps=4_000_000,
                validation_every_n_epochs=10,
                from_checkpoint=str(current),
                starting_weights=None,
                encoder_training_mode="fine_tune",
                encoder_unfreeze_step=120000,
                resolution=1024,
                reduce_ratio=0.5,
                batch_size=1,
                checkpoint_state=state,
                optimizer_metadata=optimizer_protocol_metadata(
                    model,
                    finetune.AdamWWSDConfig(),
                ),
            )

        self.assertEqual(metadata["checkpoint_sha256"], current_sha256)
        self.assertEqual(metadata["source_curriculum_step"], 40)
        self.assertEqual(
            metadata["source_curriculum_step_evidence"],
            "checkpoint_samples_seen",
        )

    def test_weights_only_branch_binds_checkpoint_curriculum_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "source.ckpt"
            torch.save({"global_step": 282200}, checkpoint)
            checkpoint_sha256 = _sha256_file(checkpoint)
            config = SimpleNamespace(data=SimpleNamespace(skip_steps=282200))

            state = finetune._validate_run_contract(
                config=config,
                from_checkpoint=None,
                starting_weights=str(checkpoint),
                max_steps=20000,
                train=True,
                protocol_version="full_page_omr_single_resize_v1",
                source_curriculum_step=282200,
                source_checkpoint_sha256=checkpoint_sha256,
            )

        self.assertEqual(state.global_step, 282200)
        self.assertEqual(state.curriculum_step, 282200)
        self.assertEqual(state.curriculum_step_source, "legacy_global_step")
        self.assertEqual(state.sha256, checkpoint_sha256)

    def test_weights_only_branch_rejects_implicit_or_mismatched_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "source.ckpt"
            torch.save(
                {
                    "global_step": 40,
                    "full_page_omr_samples_seen": 30,
                    "hyper_parameters": {"curriculum_step_offset": 10},
                },
                checkpoint,
            )
            base = dict(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=40)),
                from_checkpoint=None,
                starting_weights=str(checkpoint),
                max_steps=100,
                train=True,
                protocol_version="full_page_omr_resize_v1",
                source_checkpoint_sha256=_sha256_file(checkpoint),
            )

            with self.assertRaisesRegex(ValueError, "source_curriculum_step.*required"):
                finetune._validate_run_contract(
                    **base,
                    source_curriculum_step=None,
                )
            with self.assertRaisesRegex(ValueError, "skip_steps"):
                finetune._validate_run_contract(
                    **{**base, "config": SimpleNamespace(data=SimpleNamespace(skip_steps=39))},
                    source_curriculum_step=40,
                )
            with self.assertRaisesRegex(ValueError, "own protocol_version"):
                finetune._validate_run_contract(
                    **{**base, "protocol_version": finetune.PROTOCOL_VERSION},
                    source_curriculum_step=40,
                )
            with self.assertRaisesRegex(ValueError, "checkpoint curriculum_step"):
                finetune._validate_run_contract(
                    **{**base, "config": SimpleNamespace(data=SimpleNamespace(skip_steps=41))},
                    source_curriculum_step=41,
                )

            with self.assertRaisesRegex(ValueError, "source_checkpoint_sha256.*required"):
                finetune._validate_run_contract(
                    **{**base, "source_checkpoint_sha256": None},
                    source_curriculum_step=40,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                finetune._validate_run_contract(
                    **{**base, "source_checkpoint_sha256": "0" * 64},
                    source_curriculum_step=40,
                )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            finetune.launch(
                config_path="does-not-exist.json",
                experiment_name="test-run",
                from_checkpoint="resume.ckpt",
                starting_weights="initial.ckpt",
            )

    def test_feature_grid_is_derived_from_encoder_patch_size(self):
        self.assertEqual(finetune._feature_grid_for_resolution(1024, 16), (64, 64))
        self.assertEqual(finetune._feature_grid_for_resolution(1024, (16, 16)), (64, 64))
        for resolution, patch_size in ((0, 16), (1025, 16), (1024, (16, 32))):
            with self.subTest(resolution=resolution, patch_size=patch_size), self.assertRaises(ValueError):
                finetune._feature_grid_for_resolution(resolution, patch_size)

    def test_epoch_checkpointer_uses_requested_interval(self):
        checkpointer = finetune._build_epoch_checkpointer("test-run", "CL", 37)

        self.assertEqual(checkpointer._every_n_epochs, 37)
        self.assertEqual(checkpointer.filename, "test-run_CL-epoch")

    def test_metric_checkpointer_uses_v2_ser_and_keeps_two_candidates(self):
        checkpointer = finetune._build_metric_checkpointer("test-run", "CL")

        self.assertEqual(checkpointer.monitor, "val_SER_v2")
        self.assertEqual(checkpointer.save_top_k, 2)
        self.assertIn("{step}", checkpointer.filename)
        self.assertIn("{val_SER_v2:.4f}", checkpointer.filename)

    def test_trainer_kwargs_use_epoch_validation_without_early_stopping(self):
        callbacks = [object(), object()]
        logger = object()

        kwargs = finetune._build_trainer_kwargs(
            max_steps=4_000_000,
            validation_every_n_epochs=2_000,
            callbacks=callbacks,
            logger=logger,
        )

        self.assertEqual(kwargs["max_steps"], 4_000_000)
        self.assertEqual(kwargs["max_epochs"], 100_000)
        self.assertEqual(kwargs["check_val_every_n_epoch"], 2_000)
        self.assertEqual(kwargs["val_check_interval"], 1.0)
        self.assertEqual(kwargs["num_sanity_val_steps"], 0)
        self.assertEqual(kwargs["callbacks"], callbacks)
        self.assertEqual(len(kwargs["callbacks"]), 2)
        self.assertEqual(kwargs["precision"], "16-mixed")
        self.assertEqual(kwargs["accumulate_grad_batches"], 1)

    def test_protocol_metadata_records_metric_and_run_contract(self):
        optimizer_metadata = optimizer_protocol_metadata(
            _TinyModel(),
            finetune.AdamWWSDConfig(),
        )
        metadata = finetune._build_protocol_metadata(
            max_steps=4_000_000,
            validation_every_n_epochs=2_000,
            from_checkpoint=None,
            starting_weights=None,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            resolution=1024,
            reduce_ratio=0.5,
            batch_size=1,
            expected_training_batches_per_epoch=83,
            curriculum_steady_mixture_step=320_000,
            optimizer_metadata=optimizer_metadata,
        )

        self.assertEqual(metadata["protocol_version"], "full_page_omr_adamw_wsd_4m_v1")
        self.assertEqual(metadata["metric_version"], "canonical_v2")
        self.assertEqual(metadata["checkpoint_monitor"], "val_SER_v2")
        self.assertEqual(metadata["checkpoint_source"], "foundation")
        self.assertEqual(metadata["checkpoint_load_mode"], "fresh")
        self.assertEqual(metadata["accumulate_grad_batches"], 1)
        self.assertEqual(metadata["reduce_ratio"], 0.5)
        self.assertEqual(metadata["validation_every_n_epochs"], 2_000)
        self.assertEqual(metadata["validation_first_epoch"], 2_000)
        self.assertEqual(metadata["validation_expected_count"], 24)
        self.assertEqual(metadata["expected_training_batches_per_epoch"], 83)
        self.assertEqual(metadata["max_epochs"], 100_000)
        self.assertEqual(metadata["trainer_max_steps"], 4_000_000)
        self.assertEqual(metadata["curriculum_steady_mixture_step"], 320_000)
        self.assertEqual(metadata["optimizer"], "AdamW")
        self.assertEqual(metadata["optimizer_implementation"], "torch.optim.AdamW")
        self.assertEqual(metadata["torch_version"], str(torch.__version__))
        self.assertEqual(metadata["wsd_stable_steps"], 3_590_000)
        self.assertNotIn("learning_rate", metadata)

    def test_protocol_snapshot_is_complete_nested_and_independent(self):
        model = _TinyModel()
        metadata = finetune._build_protocol_metadata(
            max_steps=4_000_000,
            validation_every_n_epochs=2_000,
            from_checkpoint=None,
            starting_weights=None,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120_000,
            resolution=1024,
            reduce_ratio=0.5,
            batch_size=1,
            expected_training_batches_per_epoch=83,
            curriculum_steady_mixture_step=320_000,
            finetuning_technique="CL",
            attention_backend="auto",
            tokenization_mode="bekern",
            num_workers=24,
            checkpoint_every_n_epochs=100,
            optimizer_metadata=optimizer_protocol_metadata(
                model,
                finetune.AdamWWSDConfig(),
            ),
        )

        snapshot = metadata["protocol_snapshot"]
        self.assertEqual(snapshot["optimizer"]["implementation"], "torch.optim.AdamW")
        self.assertEqual(snapshot["optimizer"]["betas"], [0.9, 0.999])
        self.assertEqual(len(snapshot["optimizer"]["groups"]), 4)
        self.assertEqual(snapshot["scheduler"]["num_cycles"], 0.5)
        self.assertEqual(snapshot["scheduler"]["stable_steps"], 3_590_000)
        self.assertEqual(snapshot["trainer"]["max_steps"], 4_000_000)
        self.assertEqual(snapshot["validation"]["every_n_epochs"], 2_000)
        self.assertEqual(snapshot["curriculum"]["encoder_unfreeze_step"], 120_000)
        self.assertEqual(snapshot["metrics"]["monitor"], "val_SER_v2")
        self.assertEqual(snapshot["input"]["resolution"], 1024)

        copied = copy.deepcopy(snapshot)
        metadata["optimizer_groups"][0]["name"] = "mutated-flat-field"
        self.assertEqual(snapshot, copied)

    def test_protocol_metadata_is_archived_locally(self):
        metadata = {"protocol_version": "full_page_omr_resize_v1", "max_steps": 20000}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = finetune._write_protocol_metadata(
                "polish-resize",
                metadata,
                output_root=Path(tmpdir),
            )

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metadata)
            self.assertEqual(path.name, "protocol.json")

    def test_resize_audit_archives_shapes_and_check_images(self):
        stages = data.ResizeAuditStages(
            raw=torch.zeros(4, 6, 3, dtype=torch.uint8).numpy(),
            intermediate=torch.zeros(2, 3, 3, dtype=torch.uint8).numpy(),
            final=torch.zeros(1, 3, 8, 8),
        )
        source = Mock()
        source.resize_audit.return_value = stages
        module = SimpleNamespace(
            train_dataset=SimpleNamespace(real_source=source),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = finetune._write_resize_audit(
                module,
                experiment_name="polish-resize",
                protocol_version="full_page_omr_resize_v1",
                reduce_ratio=0.5,
                resolution=8,
                output_root=Path(tmpdir),
            )
            report = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(report["raw_shape_hwc"], [4, 6, 3])
            self.assertEqual(report["intermediate_shape_hwc"], [2, 3, 3])
            self.assertEqual(report["final_shape_nchw"], [1, 3, 8, 8])
            for filename in ("raw.png", "intermediate.png", "final.png"):
                self.assertTrue((path.parent / filename).is_file())

    def test_resize_audit_archives_grayscale_source_images(self):
        stages = data.ResizeAuditStages(
            raw=torch.zeros(4, 6, dtype=torch.uint8).numpy(),
            intermediate=torch.zeros(2, 3, dtype=torch.uint8).numpy(),
            final=torch.zeros(1, 3, 8, 8),
        )
        source = Mock()
        source.resize_audit.return_value = stages
        module = SimpleNamespace(
            train_dataset=SimpleNamespace(real_source=source),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = finetune._write_resize_audit(
                module,
                experiment_name="polish-grayscale",
                protocol_version="full_page_omr_resize_v1",
                reduce_ratio=0.5,
                resolution=8,
                output_root=Path(tmpdir),
            )
            report = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(report["raw_shape_hwc"], [4, 6])
            self.assertEqual(report["intermediate_shape_hwc"], [2, 3])
            self.assertEqual(report["raw_layout"], "HW")
            self.assertEqual(report["intermediate_layout"], "HW")
            for filename in ("raw.png", "intermediate.png", "final.png"):
                self.assertTrue((path.parent / filename).is_file())

    def test_resize_audit_is_not_applicable_to_synthetic_only_data(self):
        module = SimpleNamespace(
            train_dataset=object(),
            val_dataset=object(),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = finetune._write_resize_audit(
                module,
                experiment_name="synthetic-only",
                protocol_version=finetune.PROTOCOL_VERSION,
                reduce_ratio=1.0,
                resolution=1024,
                output_root=Path(tmpdir),
            )

        self.assertIsNone(path)

    def test_first_train_batch_callback_archives_actual_batch_shapes_once(self):
        metadata = {
            "source": "real",
            "raw_shape_hwc": [2100, 1484, 3],
            "intermediate_shape_hwc": [2100, 1484, 3],
            "final_shape_nchw": [1, 3, 1024, 1024],
        }
        batch = (
            torch.zeros((1, 3, 1024, 1024)),
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
            metadata,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "first_train_batch.json"
            callback = finetune._FirstTrainBatchInputAudit(path)
            callback.on_train_batch_start(None, None, batch, 0)
            callback.on_train_batch_start(None, None, (*batch[:3], {**metadata, "source": "other"}), 1)
            report = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(report["batch_index"], 0)
        self.assertEqual(report["source"], "real")
        self.assertEqual(report["actual_batch_shape_nchw"], [1, 3, 1024, 1024])

    @unittest.skipUnless(sys.platform == "win32", "PowerShell launcher is Windows-specific")
    def test_powershell_cairo_directory_has_no_machine_specific_default(self):
        script = (REPO_ROOT / "2.full_page_omr.ps1").read_text(encoding="utf-8")
        setting = next(
            line for line in script.splitlines()
            if "cairo_dll_directory" in line and "=" in line
        )

        self.assertEqual(setting.split("=", 1)[1].split("#", 1)[0].strip(), "$null")

    @unittest.skipUnless(sys.platform == "win32", "PowerShell launcher is Windows-specific")
    def test_powershell_dry_run_includes_default_checkpoint_interval(self):
        result = subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(REPO_ROOT / "2.full_page_omr.ps1"),
                "-DryRun",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint_every_n_epochs=100", result.stdout)
        self.assertIn("--max_steps=4000000", result.stdout)
        self.assertIn("--validation_every_n_epochs=2000", result.stdout)
        self.assertIn("--task_learning_rate=0.0001", result.stdout)
        self.assertIn("--encoder_learning_rate=0.00001", result.stdout)
        self.assertIn("--weight_decay=0.01", result.stdout)
        self.assertIn("--wsd_warmup_steps=10000", result.stdout)
        self.assertIn("--wsd_decay_steps=400000", result.stdout)
        self.assertIn("--encoder_training_mode=fine_tune", result.stdout)
        self.assertIn("--protocol_version=full_page_omr_adamw_wsd_4m_v1", result.stdout)
        self.assertNotIn("--validation_every_n_batches", result.stdout)
        self.assertIn("(num_workers=24)", result.stdout)

    def test_missing_best_checkpoint_saves_an_end_checkpoint(self):
        trainer = Mock()
        checkpointer = SimpleNamespace(best_model_path="")

        checkpoint_path = finetune._select_test_checkpoint(
            trainer,
            checkpointer,
            "test-run",
            "CL",
        )

        self.assertEqual(checkpoint_path, "weights/test-run_CL-end.ckpt")
        trainer.save_checkpoint.assert_called_once_with(checkpoint_path, weights_only=True)

    def test_checkpoint_test_failure_is_logged_and_reraised(self):
        trainer = Mock()
        trainer.test.side_effect = RuntimeError("corrupt checkpoint")
        with patch.object(finetune.logger, "exception") as log_exception:
            with self.assertRaisesRegex(RuntimeError, "corrupt checkpoint"):
                finetune._run_test(
                    trainer,
                    model_wrapper=object(),
                    data=object(),
                    checkpoint_path="weights/corrupt.ckpt",
                )

        log_exception.assert_called_once()

    def test_finetune_module_has_no_silent_exception_handlers(self):
        tree = ast.parse(inspect.getsource(finetune))
        handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]

        self.assertTrue(all(handler.type is not None for handler in handlers))
        self.assertTrue(all(not any(isinstance(statement, ast.Pass) for statement in handler.body) for handler in handlers))


class SMTPPTrainerThroughputTests(unittest.TestCase):
    def _full_checkpoint(self, module, **hyper_parameter_overrides):
        if module.protocol_snapshot is None:
            module.protocol_snapshot = _protocol_snapshot(
                module.model,
                curriculum__encoder_training_mode=module.encoder_training_mode,
                curriculum__encoder_unfreeze_step=module.encoder_unfreeze_step,
                curriculum__step_offset=module.curriculum_step_offset,
            )
            module.hparams["protocol_snapshot"] = copy.deepcopy(
                module.protocol_snapshot
            )
        configured = module.configure_optimizers()
        hyper_parameters = dict(module.hparams)
        hyper_parameters.update(hyper_parameter_overrides)
        return {
            "global_step": 0,
            "full_page_omr_samples_seen": 0,
            "hyper_parameters": hyper_parameters,
            "optimizer_states": [configured["optimizer"].state_dict()],
            "lr_schedulers": [
                configured["lr_scheduler"]["scheduler"].state_dict()
            ],
        }

    def test_configure_optimizers_uses_step_based_adamw_wsd(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )

        configured = module.configure_optimizers()

        self.assertIsInstance(configured["optimizer"], torch.optim.AdamW)
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")
        self.assertEqual(configured["lr_scheduler"]["frequency"], 1)

    def test_hyperparameters_do_not_serialize_model_object(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            batch_size=1,
            accumulate_grad_batches=1,
        )

        self.assertIn("smt_config", module.hparams)
        self.assertNotIn("smt_model", module.hparams)
        self.assertEqual(module.hparams.encoder_training_mode, "fine_tune")
        self.assertEqual(module.hparams.encoder_unfreeze_step, 120000)
        self.assertEqual(module.hparams.curriculum_step_offset, 0)
        self.assertEqual(module.hparams.batch_size, 1)
        self.assertEqual(module.hparams.accumulate_grad_batches, 1)

    def test_hparams_and_saved_checkpoint_contain_full_protocol_snapshot(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_unfreeze_step=120_000,
            protocol_snapshot=snapshot,
        )
        checkpoint = {"hyper_parameters": {}}

        module.on_save_checkpoint(checkpoint)

        self.assertEqual(module.hparams.protocol_snapshot, snapshot)
        self.assertIsNot(module.hparams.protocol_snapshot, snapshot)
        self.assertEqual(checkpoint["hyper_parameters"]["protocol_snapshot"], snapshot)
        self.assertIsNot(
            checkpoint["hyper_parameters"]["protocol_snapshot"],
            module.hparams.protocol_snapshot,
        )

    def test_fine_tune_unfreezes_encoder_at_real_data_boundary(self):
        model = _TinyModel()
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )

        module.samples_seen = 119999
        module._sync_encoder_trainability()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        module.samples_seen = 120000
        module._sync_encoder_trainability()
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_fine_tune_applies_curriculum_step_offset_before_unfreezing(self):
        model = _TinyModel()
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            curriculum_step_offset=120000,
        )

        module._sync_encoder_trainability()

        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_linear_probe_never_unfreezes_encoder(self):
        model = _TinyModel()
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=120000,
        )

        module.samples_seen = 999999
        module._sync_encoder_trainability()

        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_new_checkpoint_protocol_mismatch_is_rejected(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(
            module,
            encoder_training_mode="linear_probe",
        )

        with self.assertRaisesRegex(ValueError, "encoder_training_mode"):
            module.on_load_checkpoint(checkpoint)

    def test_full_resume_rejects_legacy_adam_checkpoint(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )
        checkpoint = {"hyper_parameters": {
            "encoder_training_mode": "fine_tune",
            "encoder_unfreeze_step": 120000,
            "curriculum_step_offset": 0,
        }}

        with self.assertRaisesRegex(ValueError, "protocol_snapshot"):
            module.on_load_checkpoint(checkpoint)

    def test_weights_only_load_may_bypass_legacy_optimizer_protocol(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
            enforce_checkpoint_protocol=False,
        )

        module.on_load_checkpoint({"hyper_parameters": {}})

    def test_full_resume_accepts_exact_adamw_wsd_identity(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module)

        module.on_load_checkpoint(checkpoint)

    def test_full_resume_retargets_changed_wsd_schedule_and_keeps_adam_state(self):
        model = _TinyModel()
        source_snapshot = _protocol_snapshot(model)
        source = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_unfreeze_step=120000,
            protocol_snapshot=source_snapshot,
        )
        checkpoint = self._full_checkpoint(source)
        global_step = 1_660_001
        checkpoint["global_step"] = global_step
        checkpoint["full_page_omr_samples_seen"] = global_step
        checkpoint["optimizer_states"][0]["state"][0] = {
            "step": torch.tensor(123.0),
        }
        source_scheduler = source.configure_optimizers()["lr_scheduler"]["scheduler"]
        source_lrs = [
            base_lr * source_scheduler.lr_lambdas[0](global_step)
            for base_lr in source_scheduler.base_lrs
        ]
        for group, current_lr in zip(
            checkpoint["optimizer_states"][0]["param_groups"],
            source_lrs,
        ):
            group["lr"] = current_lr
        source_scheduler_state = checkpoint["lr_schedulers"][0]
        source_scheduler_state["last_epoch"] = global_step
        source_scheduler_state["_step_count"] = global_step + 1
        source_scheduler_state["_last_lr"] = source_lrs

        target_snapshot = _protocol_snapshot(
            model,
            scheduler__max_steps=2_000_000,
            scheduler__stable_steps=1_650_000,
            scheduler__decay_steps=340_000,
            scheduler__min_lr_ratio=0.1,
            trainer__max_steps=2_000_000,
            validation__expected_count=12,
        )
        target = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_unfreeze_step=120000,
            max_steps=2_000_000,
            wsd_decay_steps=340_000,
            wsd_min_lr_ratio=0.1,
            protocol_snapshot=target_snapshot,
        )

        target.on_load_checkpoint(checkpoint)

        target_scheduler = target.configure_optimizers()["lr_scheduler"]["scheduler"]
        target_lrs = [
            base_lr * target_scheduler.lr_lambdas[0](global_step)
            for base_lr in target_scheduler.base_lrs
        ]
        self.assertEqual(
            checkpoint["optimizer_states"][0]["state"][0]["step"],
            torch.tensor(123.0),
        )
        self.assertEqual(
            [
                group["lr"]
                for group in checkpoint["optimizer_states"][0]["param_groups"]
            ],
            target_lrs,
        )
        self.assertEqual(checkpoint["lr_schedulers"][0]["last_epoch"], global_step)
        self.assertEqual(checkpoint["lr_schedulers"][0]["_last_lr"], target_lrs)

    def test_full_resume_uses_snapshot_instead_of_flat_compatibility_fields(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module, task_learning_rate=2e-4)

        module.on_load_checkpoint(checkpoint)

        checkpoint["hyper_parameters"]["protocol_snapshot"]["optimizer"][
            "groups"
        ][0]["learning_rate"] = 2e-5
        with self.assertRaisesRegex(ValueError, "protocol snapshot mismatch"):
            module.on_load_checkpoint(checkpoint)

    def test_full_resume_requires_one_optimizer_and_scheduler_state(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module)

        for field, value, message in (
            ("optimizer_states", [], "one AdamW optimizer state"),
            ("optimizer_states", checkpoint["optimizer_states"] * 2,
             "one AdamW optimizer state"),
            ("lr_schedulers", [], "one WSD scheduler state"),
            ("lr_schedulers", checkpoint["lr_schedulers"] * 2,
             "one WSD scheduler state"),
        ):
            with self.subTest(field=field, count=len(value)):
                malformed = {**checkpoint, field: value}
                with self.assertRaisesRegex(ValueError, message):
                    module.on_load_checkpoint(malformed)

    def test_full_resume_rejects_changed_optimizer_group_schema(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module)

        mutations = {
            "missing": lambda groups: groups.pop(),
            "reordered": lambda groups: groups.reverse(),
            "wrong_name": lambda groups: groups[0].__setitem__("name", "legacy"),
            "wrong_lr": lambda groups: groups[0].__setitem__("initial_lr", 2e-5),
            "wrong_decay": lambda groups: groups[0].__setitem__("weight_decay", 0.0),
            "trailing_scalar": lambda groups: groups.append(0),
            "bool_weight_decay": lambda groups: groups[1].__setitem__(
                "weight_decay", False
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                malformed = self._full_checkpoint(module)
                mutate(malformed["optimizer_states"][0]["param_groups"])
                with self.assertRaisesRegex(ValueError, "parameter-group mismatch"):
                    module.on_load_checkpoint(malformed)

    def test_full_resume_rejects_all_behavior_affecting_optimizer_group_mutations(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        mutations = {
            "current_lr": lambda group: group.__setitem__("lr", 123.0),
            "parameter_count": lambda group: group["params"].append(999),
            "betas": lambda group: group.__setitem__("betas", (0.8, 0.99)),
            "eps": lambda group: group.__setitem__("eps", 1e-7),
            "amsgrad": lambda group: group.__setitem__("amsgrad", True),
            "foreach": lambda group: group.__setitem__("foreach", True),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                checkpoint = self._full_checkpoint(module)
                mutate(checkpoint["optimizer_states"][0]["param_groups"][0])
                with self.assertRaisesRegex(ValueError, "parameter-group mismatch"):
                    module.on_load_checkpoint(checkpoint)

    def test_full_resume_rejects_corrupt_wsd_phase_state(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        mutations = {
            "base_lrs": lambda state: state["base_lrs"].__setitem__(0, 2e-5),
            "last_epoch": lambda state: state.__setitem__("last_epoch", 1),
            "step_count": lambda state: state.__setitem__("_step_count", 2),
            "last_lr": lambda state: state["_last_lr"].__setitem__(0, 2e-5),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                checkpoint = self._full_checkpoint(module)
                mutate(checkpoint["lr_schedulers"][0])
                with self.assertRaisesRegex(ValueError, "WSD scheduler state mismatch"):
                    module.on_load_checkpoint(checkpoint)

    def test_full_resume_accepts_native_wsd_state_in_decay_phase_without_looping(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module)
        configured = module.configure_optimizers()
        scheduler = configured["lr_scheduler"]["scheduler"]
        global_step = 3_600_001
        multiplier = scheduler.lr_lambdas[0](global_step)
        base_lrs = [1e-5, 1e-5, 1e-4, 1e-4]
        current_lrs = [base_lr * multiplier for base_lr in base_lrs]
        checkpoint["global_step"] = global_step
        checkpoint["full_page_omr_samples_seen"] = global_step
        for group, current_lr in zip(
            checkpoint["optimizer_states"][0]["param_groups"],
            current_lrs,
        ):
            group["lr"] = current_lr
        scheduler_state = checkpoint["lr_schedulers"][0]
        scheduler_state["last_epoch"] = global_step
        scheduler_state["_step_count"] = global_step + 1
        scheduler_state["_last_lr"] = current_lrs

        module.on_load_checkpoint(checkpoint)

    def test_full_resume_rejects_protocol_snapshot_mismatch(self):
        model = _TinyModel()
        snapshot = _protocol_snapshot(model)
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_unfreeze_step=120000,
            protocol_snapshot=snapshot,
        )
        mutations = {
            "optimizer_implementation": (
                ("optimizer", "implementation"),
                "another.AdamW",
            ),
            "torch_version": (("optimizer", "torch_version"), "0.0.0"),
            "betas": (("optimizer", "betas"), [0.8, 0.99]),
            "eps": (("optimizer", "eps"), 1e-7),
            "amsgrad": (("optimizer", "amsgrad"), True),
            "amsgrad_numeric_alias": (("optimizer", "amsgrad"), 0),
            "num_cycles": (("scheduler", "num_cycles"), 1.0),
            "num_cycles_bool_alias": (("scheduler", "num_cycles"), False),
        }
        for name, (path, value) in mutations.items():
            with self.subTest(name=name):
                checkpoint = self._full_checkpoint(module)
                saved_snapshot = copy.deepcopy(snapshot)
                saved_snapshot[path[0]][path[1]] = value
                checkpoint["hyper_parameters"]["protocol_snapshot"] = saved_snapshot
                with self.assertRaisesRegex(ValueError, "protocol snapshot mismatch"):
                    module.on_load_checkpoint(checkpoint)

    def test_testing_load_bypasses_training_resume_contract(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
        )
        weights_only_checkpoint = self._full_checkpoint(module)
        del weights_only_checkpoint["optimizer_states"]
        del weights_only_checkpoint["lr_schedulers"]

        module.trainer = SimpleNamespace(
            state=SimpleNamespace(fn=TrainerFn.TESTING),
        )
        module.on_load_checkpoint(weights_only_checkpoint)
        legacy_or_mismatched_weights = {
            **weights_only_checkpoint,
            "hyper_parameters": {
                **weights_only_checkpoint["hyper_parameters"],
                "encoder_training_mode": "linear_probe",
            },
        }
        module.on_load_checkpoint(legacy_or_mismatched_weights)

        unattached = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_unfreeze_step=120000,
            protocol_snapshot=module.protocol_snapshot,
        )
        with self.assertRaisesRegex(ValueError, "one AdamW optimizer state"):
            unattached.on_load_checkpoint(weights_only_checkpoint)

    def test_legacy_checkpoint_cannot_be_resumed_as_linear_probe(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=120000,
        )

        with self.assertRaisesRegex(ValueError, "protocol_snapshot"):
            module.on_load_checkpoint({"hyper_parameters": {}})

    def test_new_checkpoint_requires_unfreeze_boundary_metadata(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )
        checkpoint = self._full_checkpoint(module)
        del checkpoint["hyper_parameters"]["encoder_unfreeze_step"]

        with self.assertRaisesRegex(ValueError, "encoder_unfreeze_step"):
            module.on_load_checkpoint(checkpoint)

    def test_new_checkpoint_rejects_curriculum_offset_mismatch(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            curriculum_step_offset=120000,
        )
        checkpoint = self._full_checkpoint(module, curriculum_step_offset=0)

        with self.assertRaisesRegex(ValueError, "curriculum_step_offset"):
            module.on_load_checkpoint(checkpoint)

    def test_training_step_does_not_read_loss_scalar_each_step(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=None,
        )
        module.log = Mock()
        batch = (
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
        )

        with patch.object(torch.Tensor, "item", side_effect=AssertionError("per-step scalar sync")):
            loss = module.training_step(batch)

        self.assertIsInstance(loss, torch.Tensor)
        module.log.assert_called_once_with(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            prog_bar=True,
        )
        for name in ("best_loss", "best_image", "worst_loss", "worst_image"):
            self.assertFalse(hasattr(module, name), name)

    def test_training_step_counts_consumed_samples_not_optimizer_steps(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=None,
            batch_size=2,
            accumulate_grad_batches=4,
        )
        module.log = Mock()
        batch = (
            torch.zeros((2, 3, 2, 2)),
            torch.zeros((2, 3), dtype=torch.long),
            torch.zeros((2, 3), dtype=torch.long),
        )

        self.assertEqual(module.samples_seen, 0)
        module.training_step(batch)

        self.assertEqual(module.samples_seen, 2)
        self.assertEqual(module.curriculum_step, 2)

    def test_training_step_ignores_input_audit_metadata(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=None,
        )
        module.log = Mock()
        batch = (
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
            {"source": "real"},
        )

        loss = module.training_step(batch)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(module.samples_seen, 1)

    def test_failed_training_step_does_not_count_unconsumed_samples(self):
        model = _TinyModel()
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=None,
        )
        module.log = Mock()
        batch = (
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
        )

        with patch.object(model, "forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                module.training_step(batch)

        self.assertEqual(module.samples_seen, 0)

    def test_samples_seen_is_saved_and_restored(self):
        model = _TinyModel()
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            model,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            protocol_snapshot=_protocol_snapshot(model),
        )
        module.samples_seen = 41
        checkpoint = {}

        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["full_page_omr_samples_seen"], 41)

        restored = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )
        checkpoint.update(self._full_checkpoint(restored))
        checkpoint["full_page_omr_samples_seen"] = 41
        restored.on_load_checkpoint(checkpoint)

        self.assertEqual(restored.samples_seen, 41)

    def test_legacy_samples_seen_migration_requires_unit_batch_and_accumulation(self):
        for kwargs in ({"batch_size": 2}, {"accumulate_grad_batches": 4}):
            with self.subTest(kwargs=kwargs):
                module = SMTPP_Trainer(
                    SimpleNamespace(padding_token=0),
                    _TinyModel(),
                    encoder_training_mode="fine_tune",
                    encoder_unfreeze_step=120000,
                    **kwargs,
                )
                checkpoint = self._full_checkpoint(module)
                checkpoint["global_step"] = 73
                del checkpoint["full_page_omr_samples_seen"]

                with self.assertRaisesRegex(ValueError, "samples_seen"):
                    module.on_load_checkpoint(checkpoint)

    def test_starting_weights_do_not_import_source_samples_seen(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            enforce_checkpoint_protocol=False,
        )

        module.on_load_checkpoint({"full_page_omr_samples_seen": 999})

        self.assertEqual(module.samples_seen, 0)


if __name__ == "__main__":
    unittest.main()
