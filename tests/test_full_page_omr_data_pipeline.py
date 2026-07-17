import inspect
import multiprocessing
import pickle
import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset

from experiments.full_page_omr import data


class _UnpickleableGenerator:
    def __getstate__(self):
        raise TypeError("native generator state cannot be pickled")


class _StaticGenerator:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    def generate_full_page_score(self, **kwargs):
        self.calls += 1
        if self.failure is not None and self.calls == 1:
            raise self.failure
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return image, ["<bos>", "note", "<eos>"]


class _StaticLazyGenerator:
    def __init__(self, failure=None):
        self.instance = _StaticGenerator(failure=failure)

    def get(self):
        return self.instance


class _CountingStepCounter:
    def __init__(self, step):
        self.step = step
        self.calls = 0

    def reserve(self):
        self.calls += 1
        return self.step


class _TinySequenceDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.zeros((1, 3, 2, 2))
        sequence = torch.tensor([1, 2, 3])
        return image, sequence, sequence


class _FakeArrowRows:
    def __init__(self):
        self.samples = [
            {
                "image": np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3),
                "transcription": "score-0",
            },
            {
                "image": np.zeros((8, 10, 3), dtype=np.uint8),
                "transcription": "score-1",
            },
        ]
        self.iterations = 0
        self.row_reads = []
        self.transcription_column_reads = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        self.iterations += 1
        return iter(self.samples)

    def __getitem__(self, key):
        if key == "transcription":
            self.transcription_column_reads += 1
            return [sample["transcription"] for sample in self.samples]
        self.row_reads.append(key)
        return self.samples[key]


def _reserve_many(counter, count, output):
    output.put([counter.reserve() for _ in range(count)])


class SharedStepCounterTests(unittest.TestCase):
    def test_counter_reserves_unique_steps_and_can_reset(self):
        counter = data._SharedStepCounter(10)

        self.assertEqual([counter.reserve(), counter.reserve()], [10, 11])
        counter.reset(40)
        self.assertEqual(counter.reserve(), 40)

    def test_counter_is_shared_between_worker_processes(self):
        counter = data._SharedStepCounter(0)
        output = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(target=_reserve_many, args=(counter, 5, output))
            for _ in range(2)
        ]

        for worker in workers:
            worker.start()
        values = output.get(timeout=60) + output.get(timeout=60)
        for worker in workers:
            worker.join(timeout=60)
            self.assertFalse(worker.is_alive())
            self.assertEqual(worker.exitcode, 0)

        self.assertEqual(sorted(values), list(range(10)))


class LazyGeneratorTests(unittest.TestCase):
    def test_generator_is_created_once_per_process(self):
        owner = data._LazyVerovioGenerator("source", "train", "bekern")
        generator = object()

        with patch.object(data, "VerovioGenerator", return_value=generator) as factory:
            self.assertIs(owner.get(), generator)
            self.assertIs(owner.get(), generator)

        factory.assert_called_once_with(
            sources="source",
            split="train",
            tokenization_mode="bekern",
        )

    def test_native_generator_is_removed_when_owner_is_pickled(self):
        owner = data._LazyVerovioGenerator("source", "train", "bekern")
        with patch.object(data, "VerovioGenerator", return_value=_UnpickleableGenerator()):
            owner.get()

        restored = pickle.loads(pickle.dumps(owner))
        replacement = object()
        with patch.object(data, "VerovioGenerator", return_value=replacement) as factory:
            self.assertIs(restored.get(), replacement)

        factory.assert_called_once()


class DataLoaderPipelineTests(unittest.TestCase):
    def _make_cl_data_module(self, num_workers):
        module = data.CLFinetuningDataset.__new__(data.CLFinetuningDataset)
        LightningDataModule.__init__(module)
        module.trainer = SimpleNamespace(
            global_step=999,
            lightning_module=SimpleNamespace(samples_seen=23),
        )
        module.batch_size = 1
        module.num_workers = num_workers
        module.skip_steps = 120000
        module.step_counter = data._SharedStepCounter(0)
        module.train_dataset = _TinySequenceDataset()
        return module

    def test_train_loader_resets_counter_from_restored_samples_seen(self):
        module = self._make_cl_data_module(num_workers=0)

        module.train_dataloader()

        self.assertEqual(module.step_counter.reserve(), 120023)

    def test_synth_real_loader_resets_counter_from_restored_samples_seen(self):
        module = data.SynthRealFinetuningDataset.__new__(data.SynthRealFinetuningDataset)
        LightningDataModule.__init__(module)
        module.trainer = SimpleNamespace(
            global_step=999,
            lightning_module=SimpleNamespace(samples_seen=23),
        )
        module.batch_size = 1
        module.num_workers = 0
        module.step_counter = data._SharedStepCounter(0)
        module.train_dataset = _TinySequenceDataset()

        module.train_dataloader()

        self.assertEqual(module.step_counter.reserve(), 23)

    def test_cl_course_exposes_skip_steps_as_curriculum_offset(self):
        module = self._make_cl_data_module(num_workers=0)

        self.assertEqual(module.curriculum_step_offset, 120000)

    def test_data_courses_expose_encoder_unfreeze_boundaries(self):
        self.assertEqual(data.CL_REAL_DATA_START_STEP, 120000)
        self.assertEqual(data.SR_REAL_DATA_START_STEP, 200000)
        self.assertEqual(data.CLFinetuningDataset.encoder_unfreeze_step, 120000)
        self.assertEqual(data.SynthRealFinetuningDataset.encoder_unfreeze_step, 200000)
        self.assertIsNone(data.SyntheticGrandStaffDataset.encoder_unfreeze_step)

    def test_train_loader_enables_persistent_prefetch_workers(self):
        module = self._make_cl_data_module(num_workers=2)

        with patch.object(torch.cuda, "is_available", return_value=True):
            loader = module.train_dataloader()

        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 1)
        self.assertTrue(loader.pin_memory)
        self.assertIs(loader.worker_init_fn, data._seed_worker)

    def test_curriculum_dataset_reserves_one_step_per_sample(self):
        dataset = data.CurriculumTrainingDataset.__new__(data.CurriculumTrainingDataset)
        data.OMRIMG2SEQDataset.__init__(dataset, teacher_forcing_perc=0.0, augment=False)
        dataset.x = [np.zeros((4, 4, 3), dtype=np.uint8)]
        dataset.y = [["<bos>", "note", "<eos>"]]
        dataset.generator = _StaticLazyGenerator(
            failure=data.SyntheticScoreGenerationError("render attempts exhausted")
        )
        dataset.step_counter = _CountingStepCounter(80000)
        dataset.max_synth_prob = 0.9
        dataset.min_synth_prob = 0.2
        dataset.finetune_steps = 200000
        dataset.increase_steps = 40000
        dataset.num_cl_steps = 3
        dataset.max_cl_steps = 120000
        dataset.curriculum_stage_beginning = 2
        dataset.set_dictionaries(
            {"<pad>": 0, "<bos>": 1, "note": 2, "<eos>": 3},
            {0: "<pad>", 1: "<bos>", 2: "note", 3: "<eos>"},
        )

        with patch.object(data, "convert_img_to_tensor", return_value=torch.zeros((1, 3, 4, 4))):
            dataset[0]

        self.assertEqual(dataset.step_counter.calls, 1)
        self.assertEqual(dataset.generator.instance.calls, 2)
        self.assertFalse(hasattr(dataset, "trainer"))

    def test_worker_seed_initializes_python_and_numpy_rngs(self):
        with patch.object(torch, "initial_seed", return_value=123456):
            data._seed_worker(0)
        first = (random.random(), np.random.rand())

        with patch.object(torch, "initial_seed", return_value=123456):
            data._seed_worker(0)
        second = (random.random(), np.random.rand())

        self.assertEqual(first, second)

    def test_real_backed_datasets_keep_arrow_rows_lazy(self):
        for dataset_type in (
            data.RealDataset,
            data.CurriculumTrainingDataset,
            data.SynthToRealDataset,
        ):
            with self.subTest(dataset_type=dataset_type.__name__):
                rows = _FakeArrowRows()
                with (
                    patch.object(data, "load_dataset", return_value=rows) as load_dataset,
                    patch.object(data, "parse_kern_file", return_value=["note"]) as parse_kern,
                ):
                    dataset = dataset_type(
                        "example/dataset",
                        "train",
                        reduce_ratio=0.5,
                        augment=False,
                    )

                load_dataset.assert_called_once_with(
                    "example/dataset",
                    split="train",
                    keep_in_memory=False,
                )
                self.assertEqual(rows.iterations, 0)
                self.assertEqual(rows.row_reads, [])
                parse_kern.assert_not_called()
                self.assertFalse(isinstance(dataset.x, list))
                self.assertIs(dataset.real_source.rows, rows)

    def test_real_dataset_decodes_tokenizes_and_resizes_only_requested_row(self):
        rows = _FakeArrowRows()
        converted_images = []

        def convert(image):
            converted_images.append(image)
            return torch.zeros((1, 3, image.shape[0], image.shape[1]))

        with (
            patch.object(data, "load_dataset", return_value=rows),
            patch.object(data, "parse_kern_file", return_value=["note"]) as parse_kern,
            patch.object(data, "convert_img_to_tensor", side_effect=convert),
        ):
            dataset = data.RealDataset(
                "example/dataset",
                "train",
                teacher_forcing_perc=0.0,
                reduce_ratio=0.5,
                augment=False,
            )
            dataset.set_dictionaries(
                {"<pad>": 0, "<bos>": 1, "note": 2, "<eos>": 3},
                {0: "<pad>", 1: "<bos>", 2: "note", 3: "<eos>"},
            )

            image, decoder_input, target = dataset[0]

        self.assertEqual(rows.row_reads, [0])
        parse_kern.assert_called_once_with("score-0", tokenization_mode="bekern")
        self.assertEqual(converted_images[0].shape, (2, 3, 3))
        self.assertEqual(tuple(image.shape), (1, 3, 2, 3))
        self.assertEqual(decoder_input.tolist(), [1, 2, 3])
        self.assertEqual(target.tolist(), [1, 2, 3])

    def test_vocabulary_iteration_reads_only_arrow_transcriptions(self):
        rows = _FakeArrowRows()
        with (
            patch.object(data, "load_dataset", return_value=rows),
            patch.object(
                data,
                "parse_kern_file",
                side_effect=lambda score, tokenization_mode: [score, tokenization_mode],
            ),
        ):
            dataset = data.RealDataset("example/dataset", "train")
            sequences = list(dataset.get_gt())

        self.assertEqual(rows.transcription_column_reads, 1)
        self.assertEqual(rows.row_reads, [])
        self.assertEqual(
            sequences,
            [
                ["<bos>", "score-0", "bekern", "<eos>"],
                ["<bos>", "score-1", "bekern", "<eos>"],
            ],
        )

    def test_collate_rejects_more_than_one_image(self):
        sample = (
            torch.zeros((1, 3, 2, 2)),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
        )

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            data.batch_preparation_img2seq([sample, sample])

    def test_unknown_tokenization_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "tokenization_mode"):
            data.parse_kern_file("**kern\n*-", tokenization_mode="standard")

    def test_all_dataset_tokenization_defaults_are_bekern(self):
        dataset_types = (
            data.SyntheticOMRDataset,
            data.RealDataset,
            data.CurriculumTrainingDataset,
            data.SynthToRealDataset,
        )

        for dataset_type in dataset_types:
            with self.subTest(dataset_type=dataset_type.__name__):
                default = inspect.signature(dataset_type).parameters["tokenization_mode"].default
                self.assertEqual(default, "bekern")

    @unittest.skipUnless(sys.platform == "win32", "Windows spawn smoke test")
    def test_persistent_loader_reuses_workers_across_epochs(self):
        loader = data._build_dataloader(
            _TinySequenceDataset(),
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
        )
        try:
            self.assertEqual(len(list(loader)), 4)
            first_worker_ids = [worker.pid for worker in loader._iterator._workers]
            self.assertEqual(len(list(loader)), 4)
            second_worker_ids = [worker.pid for worker in loader._iterator._workers]
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

        self.assertEqual(first_worker_ids, second_worker_ids)


if __name__ == "__main__":
    unittest.main()
