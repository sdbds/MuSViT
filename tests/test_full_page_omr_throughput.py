import ast
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from experiments.full_page_omr import data, entrypoint, finetune
from experiments.full_page_omr import smt_trainer
from experiments.full_page_omr.smt_trainer import SMTPP_Trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.encoder = torch.nn.Linear(1, 1, bias=False)

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward(self, x, decoder_input, labels=None):
        return SimpleNamespace(loss=self.weight * 0.5)


class FullPageOMRCheckpointTests(unittest.TestCase):
    def test_public_entrypoint_forwards_checkpoint_interval_and_attention_backend(self):
        signature = inspect.signature(entrypoint.run)
        self.assertEqual(signature.parameters["checkpoint_every_n_epochs"].default, 100)
        self.assertEqual(signature.parameters["attention_backend"].default, "auto")
        self.assertEqual(signature.parameters["encoder_training_mode"].default, "fine_tune")
        self.assertEqual(signature.parameters["validation_every_n_batches"].default, 10000)
        self.assertEqual(signature.parameters["max_steps"].default, 320000)
        self.assertEqual(
            signature.parameters["protocol_version"].default,
            "full_page_omr_eval_v2",
        )

        config_path = REPO_ROOT / "experiments/full_page_omr/config/Polish_Scores/finetuning.json"
        with patch.object(entrypoint, "_launch") as launch:
            entrypoint.run(
                str(config_path),
                "test-run",
                checkpoint_every_n_epochs=37,
                attention_backend="sdpa",
                encoder_training_mode="linear_probe",
                validation_every_n_batches=25000,
                protocol_version="full_page_omr_resize_v1",
                source_curriculum_step=282200,
                source_checkpoint_sha256="a" * 64,
            )

        self.assertEqual(launch.call_args.kwargs["checkpoint_every_n_epochs"], 37)
        self.assertEqual(launch.call_args.kwargs["attention_backend"], "sdpa")
        self.assertEqual(launch.call_args.kwargs["encoder_training_mode"], "linear_probe")
        self.assertEqual(launch.call_args.kwargs["validation_every_n_batches"], 25000)
        self.assertEqual(
            launch.call_args.kwargs["protocol_version"],
            "full_page_omr_resize_v1",
        )
        self.assertEqual(launch.call_args.kwargs["source_curriculum_step"], 282200)
        self.assertEqual(launch.call_args.kwargs["source_checkpoint_sha256"], "a" * 64)

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
                    validation_every_n_batches=25000,
                    protocol_version="full_page_omr_resize_v1",
                    source_curriculum_step=282200,
                    source_checkpoint_sha256="a" * 64,
                )

        self.assertEqual(main.call_args.kwargs["checkpoint_every_n_epochs"], 37)
        self.assertEqual(main.call_args.kwargs["encoder_training_mode"], "linear_probe")
        self.assertEqual(main.call_args.kwargs["validation_every_n_batches"], 25000)
        self.assertEqual(main.call_args.kwargs["max_steps"], 320000)
        self.assertEqual(
            main.call_args.kwargs["protocol_version"],
            "full_page_omr_resize_v1",
        )
        self.assertEqual(main.call_args.kwargs["source_curriculum_step"], 282200)
        self.assertEqual(main.call_args.kwargs["source_checkpoint_sha256"], "a" * 64)

    def test_checkpoint_interval_must_be_a_positive_integer(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finetune._validate_checkpoint_every_n_epochs(value)

        self.assertEqual(finetune._validate_checkpoint_every_n_epochs(37), 37)

    def test_validation_interval_and_production_max_steps_are_validated(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                finetune._validate_validation_every_n_batches(value)
        self.assertEqual(finetune._validate_validation_every_n_batches(10000), 10000)

        for value in (0, -1, True, 1.5):
            with self.subTest(max_steps=value), self.assertRaises(ValueError):
                finetune._validate_max_steps(value, train=True)
        self.assertEqual(finetune._validate_max_steps(320000, train=True), 320000)
        self.assertEqual(finetune._validate_max_steps(-1, train=False), -1)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "resume.ckpt"
            torch.save({"global_step": 320000}, checkpoint)

            with self.assertRaisesRegex(ValueError, "max_steps.*global_step"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=0)),
                    from_checkpoint=str(checkpoint),
                    starting_weights=None,
                    max_steps=320000,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                )

    def test_full_resume_rejects_curriculum_offset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = Path(tmpdir) / "legacy.ckpt"
            torch.save({"global_step": 40}, legacy)
            with self.assertRaisesRegex(ValueError, "legacy.*skip_steps=0"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=10)),
                    from_checkpoint=str(legacy),
                    starting_weights=None,
                    max_steps=100,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                )

            current = Path(tmpdir) / "current.ckpt"
            torch.save(
                {
                    "global_step": 40,
                    "full_page_omr_samples_seen": 30,
                    "hyper_parameters": {"curriculum_step_offset": 10},
                },
                current,
            )
            with self.assertRaisesRegex(ValueError, "curriculum_step_offset"):
                finetune._validate_run_contract(
                    config=SimpleNamespace(data=SimpleNamespace(skip_steps=9)),
                    from_checkpoint=str(current),
                    starting_weights=None,
                    max_steps=100,
                    train=True,
                    protocol_version=finetune.PROTOCOL_VERSION,
                    source_curriculum_step=None,
                    source_checkpoint_sha256=None,
                )

            state = finetune._validate_run_contract(
                config=SimpleNamespace(data=SimpleNamespace(skip_steps=10)),
                from_checkpoint=str(current),
                starting_weights=None,
                max_steps=100,
                train=True,
                protocol_version=finetune.PROTOCOL_VERSION,
                source_curriculum_step=None,
                source_checkpoint_sha256=None,
            )

        self.assertEqual(state.curriculum_step_offset, 10)

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

    def test_trainer_kwargs_use_step_based_validation_without_early_stopping(self):
        callbacks = [object(), object()]
        logger = object()

        kwargs = finetune._build_trainer_kwargs(
            max_steps=320000,
            validation_every_n_batches=10000,
            callbacks=callbacks,
            logger=logger,
        )

        self.assertEqual(kwargs["max_steps"], 320000)
        self.assertIsNone(kwargs["check_val_every_n_epoch"])
        self.assertEqual(kwargs["val_check_interval"], 10000)
        self.assertEqual(kwargs["callbacks"], callbacks)
        self.assertEqual(len(kwargs["callbacks"]), 2)
        self.assertEqual(kwargs["precision"], "16-mixed")
        self.assertEqual(kwargs["accumulate_grad_batches"], 1)

    def test_protocol_metadata_records_metric_and_run_contract(self):
        metadata = finetune._build_protocol_metadata(
            max_steps=320000,
            validation_every_n_batches=10000,
            from_checkpoint=None,
            starting_weights=None,
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
            resolution=1024,
            reduce_ratio=0.5,
            batch_size=1,
        )

        self.assertEqual(metadata["protocol_version"], "full_page_omr_eval_v2")
        self.assertEqual(metadata["metric_version"], "canonical_v2")
        self.assertEqual(metadata["checkpoint_monitor"], "val_SER_v2")
        self.assertEqual(metadata["checkpoint_source"], "foundation")
        self.assertEqual(metadata["checkpoint_load_mode"], "fresh")
        self.assertEqual(metadata["accumulate_grad_batches"], 1)
        self.assertEqual(metadata["reduce_ratio"], 0.5)

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
        self.assertIn("--validation_every_n_batches=10000", result.stdout)
        self.assertIn("--max_steps=320000", result.stdout)
        self.assertIn("--encoder_training_mode=fine_tune", result.stdout)
        self.assertIn("--protocol_version=full_page_omr_eval_v2", result.stdout)
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
        checkpoint = {
            "hyper_parameters": {
                "encoder_training_mode": "linear_probe",
                "encoder_unfreeze_step": 120000,
            }
        }

        with self.assertRaisesRegex(ValueError, "encoder_training_mode"):
            module.on_load_checkpoint(checkpoint)

    def test_legacy_checkpoint_uses_explicit_protocol_with_warning(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )

        with patch.object(smt_trainer.logger, "warning") as warning:
            module.on_load_checkpoint({"hyper_parameters": {}, "global_step": 73})

        self.assertEqual(module.samples_seen, 73)
        self.assertEqual(warning.call_count, 2)

    def test_legacy_checkpoint_cannot_be_resumed_as_linear_probe(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="linear_probe",
            encoder_unfreeze_step=120000,
        )

        with self.assertRaisesRegex(ValueError, "legacy.*fine_tune"):
            module.on_load_checkpoint({"hyper_parameters": {}})

    def test_new_checkpoint_requires_unfreeze_boundary_metadata(self):
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
        )
        checkpoint = {"hyper_parameters": {"encoder_training_mode": "fine_tune"}}

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
        checkpoint = {
            "hyper_parameters": {
                "encoder_training_mode": "fine_tune",
                "encoder_unfreeze_step": 120000,
                "curriculum_step_offset": 0,
            }
        }

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
        module = SMTPP_Trainer(
            SimpleNamespace(padding_token=0),
            _TinyModel(),
            encoder_training_mode="fine_tune",
            encoder_unfreeze_step=120000,
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
        checkpoint["hyper_parameters"] = {
            "encoder_training_mode": "fine_tune",
            "encoder_unfreeze_step": 120000,
            "curriculum_step_offset": 0,
        }
        restored.on_load_checkpoint(checkpoint)

        self.assertEqual(restored.samples_seen, 41)

    def test_legacy_samples_seen_migration_requires_unit_batch_and_accumulation(self):
        checkpoint = {
            "global_step": 73,
            "hyper_parameters": {
                "encoder_training_mode": "fine_tune",
                "encoder_unfreeze_step": 120000,
                "curriculum_step_offset": 0,
            },
        }
        for kwargs in ({"batch_size": 2}, {"accumulate_grad_batches": 4}):
            with self.subTest(kwargs=kwargs):
                module = SMTPP_Trainer(
                    SimpleNamespace(padding_token=0),
                    _TinyModel(),
                    encoder_training_mode="fine_tune",
                    encoder_unfreeze_step=120000,
                    **kwargs,
                )

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
