import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from experiments.full_page_omr import benchmark_generation as benchmark
from experiments.full_page_omr.smt_foundation.modeling_smt import (
    SMTFoundationModelForCausalLM,
)


class LockedBenchmarkContractTests(unittest.TestCase):
    def test_reference_identity_is_hard_coded(self):
        locked = benchmark.locked_inputs()

        self.assertEqual(locked["gpu"]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(
            locked["gpu"]["uuid"],
            "GPU-a70be80e-9cef-95c2-7557-52448110b38e",
        )
        self.assertEqual(locked["dtype"], "torch.float16")
        self.assertEqual(
            locked["checkpoint"]["sha256"],
            "ebfbcb14b7bb6fad60280939341c7b603d3713b3d7f67cd9cae3c44b8f161d0c",
        )
        self.assertEqual(locked["checkpoint"]["global_step"], 282200)
        self.assertEqual(
            locked["dataset"]["revision"],
            "b3170c8b8f322885b566efe9e264af9328b5603f",
        )
        self.assertEqual(locked["validation"]["rows"], list(range(10)))
        self.assertEqual(locked["validation"]["reduce_ratio"], 0.5)
        self.assertEqual(locked["validation"]["resolution"], 1024)
        self.assertEqual(locked["validation"]["maxlen"], 7512)
        self.assertEqual(locked["validation"]["attention_backend"], "auto")
        self.assertEqual(locked["prefix_lengths"], [1024, 2048, 4096])

    def test_incremental_generation_is_not_the_ungated_default(self):
        default = inspect.signature(
            SMTFoundationModelForCausalLM.generate_token_ids
        ).parameters["use_incremental"].default

        self.assertIs(default, False)

    def test_prefix_starts_with_bos_and_cycles_only_content_tokens(self):
        prefix = benchmark.build_cycled_prefix(
            [1, 7, 8, 2, 0, 1],
            length=8,
            bos_id=1,
            eos_id=2,
            pad_id=0,
        )

        self.assertEqual(prefix, (1, 7, 8, 7, 8, 7, 8, 7))

    def test_prefix_rejects_empty_content_and_invalid_length(self):
        with self.assertRaisesRegex(ValueError, "content"):
            benchmark.build_cycled_prefix(
                [1, 2, 0],
                length=4,
                bos_id=1,
                eos_id=2,
                pad_id=0,
            )

    def test_gpu_isolation_uses_uuid_not_driver_index_order(self):
        nvidia_output = (
            "7, NVIDIA GeForce RTX 4090, "
            "GPU-a70be80e-9cef-95c2-7557-52448110b38e, 581.57, 63, 0\n"
        )
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            with (
                patch.object(benchmark, "_run_nvidia_smi", return_value=nvidia_output),
                patch.object(torch.cuda, "is_initialized", return_value=False),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
                patch.object(torch.cuda, "get_device_name", return_value=benchmark.GPU_NAME),
            ):
                benchmark._LockedBenchmarkRuntime().ensure_reference_gpu()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], benchmark.GPU_UUID)
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        with self.assertRaisesRegex(ValueError, "length"):
            benchmark.build_cycled_prefix(
                [1, 7, 2],
                length=0,
                bos_id=1,
                eos_id=2,
                pad_id=0,
            )


class TimingProtocolTests(unittest.TestCase):
    def test_microbenchmark_uses_three_warmups_and_ten_timed_samples(self):
        function = Mock(side_effect=lambda: function.call_count)
        synchronize = Mock()
        clock_values = []
        for index in range(10):
            clock_values.extend([float(index * 2), float(index * 2 + index + 1)])
        clock = Mock(side_effect=clock_values)

        summary, results = benchmark.measure_microbenchmark(
            function,
            synchronize=synchronize,
            clock=clock,
        )

        self.assertEqual(function.call_count, 13)
        self.assertEqual(synchronize.call_count, 23)
        self.assertEqual(clock.call_count, 20)
        self.assertEqual(summary["warmups"], 3)
        self.assertEqual(summary["sample_count"], 10)
        self.assertEqual(summary["seconds"], [float(index + 1) for index in range(10)])
        self.assertEqual(summary["median_seconds"], 5.5)
        self.assertEqual(len(results), 10)

    def test_full_validation_uses_one_warmup_and_three_timed_samples(self):
        function = Mock(return_value=("tokens",))
        synchronize = Mock()
        clock = Mock(side_effect=[0.0, 3.0, 10.0, 12.0, 20.0, 21.0])

        summary, results = benchmark.measure_full_validation(
            function,
            synchronize=synchronize,
            clock=clock,
        )

        self.assertEqual(function.call_count, 4)
        self.assertEqual(synchronize.call_count, 7)
        self.assertEqual(summary["warmups"], 1)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["seconds"], [3.0, 2.0, 1.0])
        self.assertEqual(summary["median_seconds"], 2.0)
        self.assertEqual(results, [("tokens",)] * 3)


class BenchmarkGateTests(unittest.TestCase):
    def test_all_required_performance_and_correctness_gates_are_explicit(self):
        micro = {
            "1024": {
                "uncached": {"median_seconds": 1.0},
                "incremental": {"median_seconds": 1.0},
            },
            "2048": {
                "uncached": {"median_seconds": 4.0},
                "incremental": {"median_seconds": 2.0},
            },
        }
        full_validation = {
            "uncached": {"median_seconds": 100.0},
            "incremental": {"median_seconds": 90.0},
            "token_sequences_equal": True,
        }

        gates = benchmark.evaluate_gates(
            micro,
            full_validation,
            memory_contract_passed=True,
            numerical_passed=True,
        )

        self.assertTrue(gates["micro_1024_not_slower"])
        self.assertTrue(gates["micro_2048_at_least_2x"])
        self.assertTrue(gates["full_validation_at_most_90_percent"])
        self.assertTrue(gates["token_sequences_equal"])
        self.assertTrue(gates["memory_contract"])
        self.assertTrue(gates["micro_prefix_argmax_equal"])
        self.assertTrue(gates["passed"])

    def test_boundary_failure_keeps_the_gate_closed(self):
        micro = {
            "1024": {
                "uncached": {"median_seconds": 1.0},
                "incremental": {"median_seconds": 1.01},
            },
            "2048": {
                "uncached": {"median_seconds": 4.0},
                "incremental": {"median_seconds": 2.01},
            },
        }
        full_validation = {
            "uncached": {"median_seconds": 100.0},
            "incremental": {"median_seconds": 90.01},
            "token_sequences_equal": False,
        }

        gates = benchmark.evaluate_gates(
            micro,
            full_validation,
            memory_contract_passed=False,
            numerical_passed=False,
        )

        self.assertFalse(gates["passed"])

    def test_unavailable_reference_writes_not_run_without_calling_runner(self):
        runtime = Mock()
        runtime.ensure_reference_gpu.side_effect = benchmark.BenchmarkUnavailable(
            "reference GPU is busy"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "report.json"
            with patch.object(benchmark, "_LockedBenchmarkRuntime", return_value=runtime):
                report = benchmark.main("missing.ckpt", output)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "not-run")
        self.assertIn("busy", report["reason"])
        self.assertEqual(written, report)
        runtime.run.assert_not_called()

    def test_completed_measurement_status_reflects_the_gate(self):
        runtime = Mock()
        runtime.ensure_reference_gpu.return_value = {"name": "GPU", "uuid": "uuid"}
        runtime.run.return_value = {
            "environment": {},
            "identity": {},
            "decoder_microbenchmark": {
                "1024": {
                    "uncached": {"median_seconds": 1.0},
                    "incremental": {"median_seconds": 0.5},
                },
                "2048": {
                    "uncached": {"median_seconds": 4.0},
                    "incremental": {"median_seconds": 1.0},
                },
            },
            "full_validation": {
                "uncached": {"median_seconds": 100.0},
                "incremental": {"median_seconds": 80.0},
                "token_sequences_equal": True,
            },
            "memory_contract": {"passed": True},
            "numerical": {
                "micro_max_logits_abs_diff": 0.0,
                "all_prefix_argmax_equal": True,
            },
        }
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "report.json"
            with patch.object(benchmark, "_LockedBenchmarkRuntime", return_value=runtime):
                report = benchmark.main("reference.ckpt", output)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["gates"]["passed"])
        runtime.run.assert_called_once_with(
            "reference.ckpt",
            {"name": "GPU", "uuid": "uuid"},
        )


if __name__ == "__main__":
    unittest.main()
