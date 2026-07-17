import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from experiments.full_page_omr import diagnose_checkpoints as diagnostics


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_entry(path, *, epoch, global_step, **source):
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "epoch": epoch,
        "global_step": global_step,
        **source,
    }


def _manifest(post, foundation_digest, *, mode="post-only", pre=None):
    checkpoints = {"post": post}
    if pre is not None:
        checkpoints["pre"] = pre
    return {
        "mode": mode,
        "foundation": {
            "model_id": diagnostics.FOUNDATION_MODEL_ID,
            "revision": diagnostics.FOUNDATION_REVISION,
            "encoder_state_sha256": foundation_digest,
        },
        "dataset": {
            "id": diagnostics.DATASET_ID,
            "revision": diagnostics.DATASET_REVISION,
        },
        "validation": dict(diagnostics.FIXED_VALIDATION_PROTOCOL),
        "runtime": {
            "resolved_attention_backend": "sdpa",
            "generation_path": "full-prefix",
            "gpu_uuid": "GPU-test",
            "software_versions": {"torch": torch.__version__, "test": "1"},
        },
        "checkpoints": checkpoints,
    }


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="ascii")
    return path


def _evaluation(role, runtime):
    value = 2.0 if role == "post" else 1.0
    pages = [
        {
            "row": row,
            "CER_v2": value,
            "SER_v2": value + 1.0,
            "LER_v2": value + 2.0,
            "terminated_by_eos": row != 9,
            "truncated": row == 9,
            "decode_seconds": 0.1,
        }
        for row in range(10)
    ]
    return {
        "pages": pages,
        "aggregate": {
            "CER_v2": value,
            "SER_v2": value + 1.0,
            "LER_v2": value + 2.0,
        },
        "eos_terminated": 9,
        "truncated": 1,
        "decode_seconds": 1.0,
        "environment": dict(runtime),
    }


class CheckpointManifestTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.reference = {"encoder.layer.0.weight": torch.tensor([3.0, 4.0])}
        self.foundation_digest = diagnostics.encoder_state_sha256(self.reference)
        self.post_path = self.root / "post.ckpt"
        torch.save({"epoch": 3500, "global_step": 282200}, self.post_path)
        self.post = _checkpoint_entry(
            self.post_path,
            epoch=3500,
            global_step=282200,
            source_note="legacy local checkpoint",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_load_manifest_requires_fixed_identity_and_protocol(self):
        manifest = _manifest(self.post, self.foundation_digest)
        path = _write_manifest(self.root / "manifest.json", manifest)

        loaded = diagnostics.load_manifest(path)

        self.assertEqual(loaded, manifest)

        mutations = {
            "foundation revision": lambda item: item["foundation"].update(revision="main"),
            "dataset revision": lambda item: item["dataset"].update(revision="main"),
            "validation rows": lambda item: item["validation"].update(rows=list(range(9))),
            "checkpoint source": lambda item: (
                item["checkpoints"]["post"].pop("source_note"),
                item["checkpoints"]["post"].pop("run_id", None),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = json.loads(json.dumps(manifest))
                mutate(candidate)
                _write_manifest(path, candidate)
                with self.assertRaises(diagnostics.ManifestError):
                    diagnostics.load_manifest(path)

    def test_pre_post_mode_requires_a_real_pre_unfreeze_checkpoint(self):
        manifest = _manifest(
            self.post,
            self.foundation_digest,
            mode="pre-post",
        )
        path = _write_manifest(self.root / "manifest.json", manifest)

        with self.assertRaisesRegex(diagnostics.ManifestError, "pre"):
            diagnostics.load_manifest(path)

    def test_post_only_mode_without_pre_checkpoint_is_partial(self):
        manifest = _manifest(self.post, self.foundation_digest)
        loaded = diagnostics.load_manifest(
            _write_manifest(self.root / "manifest.json", manifest)
        )

        report = diagnostics.run_diagnostics(
            loaded,
            load_foundation_encoder=lambda _: self.reference,
            load_checkpoint_encoder=lambda identity, _: {
                "encoder.layer.0.weight": torch.tensor([0.0, 0.0])
            },
            evaluate_checkpoint=lambda identity, item: _evaluation(
                identity["role"], item["runtime"]
            ),
        )

        self.assertEqual(report["status"], "partial")
        self.assertNotIn("comparison", report)
        self.assertEqual(report["encoder_drift"]["post"]["global"], 1.0)

    def test_hash_mismatch_fails_before_checkpoint_deserialization(self):
        declared = dict(self.post, sha256="0" * 64)

        with patch.object(torch, "load", side_effect=AssertionError("must not load")) as load:
            with self.assertRaisesRegex(diagnostics.CheckpointIdentityError, "SHA-256"):
                diagnostics.verify_checkpoint_identity(declared, role="post")

        load.assert_not_called()

    def test_epoch_or_step_mismatch_is_rejected(self):
        for field, value in (("epoch", 3499), ("global_step", 282199)):
            with self.subTest(field=field):
                declared = dict(self.post, **{field: value})
                with self.assertRaisesRegex(diagnostics.CheckpointIdentityError, field):
                    diagnostics.verify_checkpoint_identity(declared, role="post")

    def test_all_checkpoint_identities_are_verified_before_model_loading(self):
        pre_path = self.root / "pre.ckpt"
        torch.save({"epoch": 100, "global_step": 119999}, pre_path)
        pre = _checkpoint_entry(
            pre_path,
            epoch=100,
            global_step=119999,
            run_id="wandb/pre",
        )
        manifest = _manifest(
            dict(self.post, sha256="f" * 64),
            self.foundation_digest,
            mode="pre-post",
            pre=pre,
        )
        model_loader = Mock()
        evaluator = Mock()

        with self.assertRaises(diagnostics.CheckpointIdentityError):
            diagnostics.run_diagnostics(
                manifest,
                load_foundation_encoder=lambda _: self.reference,
                load_checkpoint_encoder=model_loader,
                evaluate_checkpoint=evaluator,
            )

        model_loader.assert_not_called()
        evaluator.assert_not_called()


class EncoderStateTests(unittest.TestCase):
    def test_state_digest_is_sorted_and_covers_name_dtype_shape_and_bytes(self):
        first = {
            "z.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "a.bias": torch.tensor([3.0], dtype=torch.float32),
        }
        reordered = {"a.bias": first["a.bias"], "z.weight": first["z.weight"]}

        self.assertEqual(
            diagnostics.encoder_state_sha256(first),
            diagnostics.encoder_state_sha256(reordered),
        )

        variants = (
            {"z.weight": first["z.weight"], "renamed": first["a.bias"]},
            {"z.weight": first["z.weight"].to(torch.float64), "a.bias": first["a.bias"]},
            {"z.weight": first["z.weight"].reshape(2, 1), "a.bias": first["a.bias"]},
            {"z.weight": first["z.weight"] + 1, "a.bias": first["a.bias"]},
        )
        digest = diagnostics.encoder_state_sha256(first)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(digest, diagnostics.encoder_state_sha256(variant))

    def test_relative_drift_uses_float64_and_reports_transformer_blocks(self):
        reference = {
            "encoder.layer.0.weight": torch.tensor([3.0, 4.0], dtype=torch.float16),
            "encoder.layer.1.weight": torch.tensor([0.0, 12.0], dtype=torch.float16),
        }
        candidate = {
            "encoder.layer.0.weight": torch.tensor([0.0, 0.0], dtype=torch.float16),
            "encoder.layer.1.weight": torch.tensor([0.0, 12.0], dtype=torch.float16),
        }

        result = diagnostics.relative_encoder_drift(reference, candidate)

        self.assertAlmostEqual(result["global"], 5.0 / 13.0)
        self.assertEqual(result["blocks"]["encoder.layer.0"], 1.0)
        self.assertEqual(result["blocks"]["encoder.layer.1"], 0.0)

    def test_relative_drift_rejects_key_and_shape_mismatches(self):
        reference = {"block.weight": torch.ones(2)}
        candidates = (
            {},
            {"block.weight": torch.ones(2), "extra": torch.ones(1)},
            {"block.weight": torch.ones(1, 2)},
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with self.assertRaises(diagnostics.EncoderStateError):
                    diagnostics.relative_encoder_drift(reference, candidate)


class DiagnosticReportTests(unittest.TestCase):
    def test_main_writes_the_validated_structured_report(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            checkpoint_path = root / "post.ckpt"
            torch.save({"epoch": 1, "global_step": 120000}, checkpoint_path)
            reference = {"weight": torch.ones(1)}
            manifest = _manifest(
                _checkpoint_entry(
                    checkpoint_path,
                    epoch=1,
                    global_step=120000,
                    source_note="test",
                ),
                diagnostics.encoder_state_sha256(reference),
            )
            manifest_path = _write_manifest(root / "manifest.json", manifest)
            output_path = root / "report.json"
            runtime = Mock()
            runtime.load_foundation_encoder.side_effect = lambda _: reference
            runtime.load_checkpoint_encoder.side_effect = lambda *_: reference
            runtime.evaluate_checkpoint.side_effect = lambda identity, item: _evaluation(
                identity["role"], item["runtime"]
            )

            with patch.object(diagnostics, "_DefaultDiagnosticRuntime", return_value=runtime):
                report = diagnostics.main(manifest_path, output_path)

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "partial")
        self.assertEqual(written, report)

    def test_pre_post_report_contains_page_deltas_and_exact_input_manifest(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            pre_path = root / "pre.ckpt"
            post_path = root / "post.ckpt"
            torch.save({"epoch": 100, "global_step": 119999}, pre_path)
            torch.save({"epoch": 3500, "global_step": 282200}, post_path)
            reference = {"encoder.layer.0.weight": torch.tensor([3.0, 4.0])}
            manifest = _manifest(
                _checkpoint_entry(
                    post_path,
                    epoch=3500,
                    global_step=282200,
                    run_id="wandb/post",
                ),
                diagnostics.encoder_state_sha256(reference),
                mode="pre-post",
                pre=_checkpoint_entry(
                    pre_path,
                    epoch=100,
                    global_step=119999,
                    run_id="wandb/pre",
                ),
            )

            report = diagnostics.run_diagnostics(
                manifest,
                load_foundation_encoder=lambda _: reference,
                load_checkpoint_encoder=lambda identity, _: (
                    reference
                    if identity["role"] == "pre"
                    else {"encoder.layer.0.weight": torch.tensor([0.0, 0.0])}
                ),
                evaluate_checkpoint=lambda identity, item: _evaluation(
                    identity["role"], item["runtime"]
                ),
            )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["manifest"], manifest)
        self.assertEqual(report["checkpoints"]["post"]["global_step"], 282200)
        self.assertEqual(report["evaluations"]["post"]["eos_terminated"], 9)
        self.assertEqual(report["evaluations"]["post"]["truncated"], 1)
        self.assertEqual(report["comparison"]["aggregate_delta"]["SER_v2"], 1.0)
        self.assertEqual(report["comparison"]["pages"][0]["SER_v2_delta"], 1.0)

    def test_foundation_digest_mismatch_fails_before_checkpoint_model_loading(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "post.ckpt"
            torch.save({"epoch": 1, "global_step": 120000}, path)
            manifest = _manifest(
                _checkpoint_entry(
                    path,
                    epoch=1,
                    global_step=120000,
                    source_note="test",
                ),
                "0" * 64,
            )
            candidate_loader = Mock()

            with self.assertRaisesRegex(diagnostics.EncoderStateError, "foundation"):
                diagnostics.run_diagnostics(
                    manifest,
                    load_foundation_encoder=lambda _: {"weight": torch.ones(1)},
                    load_checkpoint_encoder=candidate_loader,
                    evaluate_checkpoint=Mock(),
                )

        candidate_loader.assert_not_called()

    def test_evaluation_environment_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "post.ckpt"
            torch.save({"epoch": 1, "global_step": 120000}, path)
            reference = {"weight": torch.ones(1)}
            manifest = _manifest(
                _checkpoint_entry(
                    path,
                    epoch=1,
                    global_step=120000,
                    source_note="test",
                ),
                diagnostics.encoder_state_sha256(reference),
            )
            result = _evaluation("post", manifest["runtime"])
            result["environment"]["gpu_uuid"] = "GPU-other"

            with self.assertRaisesRegex(diagnostics.DiagnosticError, "environment"):
                diagnostics.run_diagnostics(
                    manifest,
                    load_foundation_encoder=lambda _: reference,
                    load_checkpoint_encoder=lambda *_: reference,
                    evaluate_checkpoint=lambda *_: result,
                )


class DefaultDiagnosticRuntimeTests(unittest.TestCase):
    def test_foundation_snapshot_prefers_the_exact_cached_revision(self):
        download = Mock(return_value="cached-snapshot")

        snapshot = diagnostics._DefaultDiagnosticRuntime._download_foundation_snapshot(
            download
        )

        self.assertEqual(snapshot, "cached-snapshot")
        download.assert_called_once_with(
            repo_id=diagnostics.FOUNDATION_MODEL_ID,
            revision=diagnostics.FOUNDATION_REVISION,
            allow_patterns=["config.json", "model.safetensors"],
            local_files_only=True,
        )

    def test_gpu_uuid_comes_from_the_actual_torch_device(self):
        properties = SimpleNamespace(uuid="a70be80e-9cef-95c2-7557-52448110b38e")
        with (
            patch.object(torch.cuda, "get_device_properties", return_value=properties),
            patch.object(
                diagnostics.subprocess,
                "run",
                side_effect=AssertionError("driver index must not identify CUDA device"),
            ),
        ):
            uuid = diagnostics._DefaultDiagnosticRuntime._visible_gpu_uuid()

        self.assertEqual(uuid, "GPU-a70be80e-9cef-95c2-7557-52448110b38e")


if __name__ == "__main__":
    unittest.main()
