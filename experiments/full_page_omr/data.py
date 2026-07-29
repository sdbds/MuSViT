import cv2
import torch
import random
import multiprocessing
import numpy as np
from dataclasses import dataclass
from .config.ExperimentConfigWrapper import ExperimentConfig
from .batching import batch_preparation_img2seq
from .Generator.SynthGenerator import VerovioGenerator, SyntheticScoreGenerationError
from .data_augmentation.data_augmentation import augment, convert_img_to_tensor
from .tokenization import (
    clean_kern,
    parse_kern_file,
    validate_tokenization_mode,
)
from .utils.vocab_utils import check_and_retrieveVocabulary

from datasets import load_dataset
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule


CL_CURRICULUM_STAGE_STEPS = 40000
CL_SYNTHETIC_STAGES = 3
CL_REAL_DATA_START_STEP = CL_CURRICULUM_STAGE_STEPS * CL_SYNTHETIC_STAGES
SR_REAL_DATA_START_STEP = 200000


@dataclass(frozen=True)
class ResizeAuditStages:
    raw: np.ndarray
    intermediate: np.ndarray
    final: torch.Tensor


class _SharedStepCounter:
    def __init__(self, initial_step: int = 0) -> None:
        self._value = multiprocessing.Value("q", int(initial_step), lock=True)

    def reset(self, step: int) -> None:
        with self._value.get_lock():
            self._value.value = int(step)

    def reserve(self) -> int:
        with self._value.get_lock():
            step = self._value.value
            self._value.value += 1
        return step


class _LazyVerovioGenerator:
    def __init__(self, sources: str, split: str, tokenization_mode: str) -> None:
        self.sources = sources
        self.split = split
        self.tokenization_mode = validate_tokenization_mode(tokenization_mode)
        self._generator = None

    def get(self) -> VerovioGenerator:
        if self._generator is None:
            self._generator = VerovioGenerator(
                sources=self.sources,
                split=self.split,
                tokenization_mode=self.tokenization_mode,
            )
        return self._generator

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generator"] = None
        return state


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_dataloader(dataset: Dataset, batch_size: int, num_workers: int,
                      *, shuffle: bool = False, persistent_workers: bool = False):
    if batch_size != 1:
        raise ValueError("full-page OMR requires batch_size=1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "collate_fn": batch_preparation_img2seq,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update(
            worker_init_fn=_seed_worker,
            prefetch_factor=1,
            persistent_workers=persistent_workers,
        )
    return torch.utils.data.DataLoader(**kwargs)


def _retry_synthetic_sample(generate, max_attempts: int = 5):
    for attempt in range(max_attempts):
        try:
            return generate()
        except SyntheticScoreGenerationError:
            if attempt == max_attempts - 1:
                raise

class _ArrowOMRSource:
    def __init__(self, dataset_ref: str, split: str, tokenization_mode: str,
                 reduce_ratio: float) -> None:
        self.tokenization_mode = validate_tokenization_mode(tokenization_mode)
        if (
            isinstance(reduce_ratio, bool)
            or not isinstance(reduce_ratio, (int, float))
            or reduce_ratio <= 0
        ):
            raise ValueError("reduce_ratio must be a positive number")
        self.reduce_ratio = float(reduce_ratio)
        self.rows = load_dataset(
            dataset_ref,
            split=split,
            keep_in_memory=False,
        )

    def __len__(self):
        return len(self.rows)

    def _tokenize(self, transcription):
        return [
            '<bos>',
            *parse_kern_file(
                transcription,
                tokenization_mode=self.tokenization_mode,
            ),
            '<eos>',
        ]

    def _image_stages(self, index):
        sample = self.rows[index]
        raw = np.asarray(sample['image'])
        intermediate = raw
        if self.reduce_ratio != 1.0:
            width = int(np.ceil(raw.shape[1] * self.reduce_ratio))
            height = int(np.ceil(raw.shape[0] * self.reduce_ratio))
            intermediate = cv2.resize(raw, (width, height))
        return sample, raw, intermediate

    def __getitem__(self, index):
        sample, _, intermediate = self._image_stages(index)
        return intermediate, self._tokenize(sample["transcription"])

    def get_with_resize_metadata(self, index):
        sample, raw, intermediate = self._image_stages(index)
        return (
            intermediate,
            self._tokenize(sample["transcription"]),
            {
                "source": "real",
                "raw_shape_hwc": list(raw.shape),
                "intermediate_shape_hwc": list(intermediate.shape),
            },
        )

    def resize_audit(self, index) -> ResizeAuditStages:
        _, raw, intermediate = self._image_stages(index)
        return ResizeAuditStages(
            raw=raw,
            intermediate=intermediate,
            final=convert_img_to_tensor(intermediate),
        )

    def iter_token_sequences(self):
        for transcription in self.rows["transcription"]:
            yield self._tokenize(transcription)


class OMRIMG2SEQDataset(Dataset):
    def __init__(self, teacher_forcing_perc=0.2, augment=False) -> None:
        self.x = None
        self.y = None
        self.teacher_forcing_error_rate = teacher_forcing_perc
        self.augment = augment

        super().__init__()
    
    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))
        
        return errored_sequence

    def __len__(self):
        return len(self.x)

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width
    
    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']
    
    def get_dictionaries(self):
        return self.w2i, self.i2w
    
    def get_i2w(self):
        return self.i2w

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split="train", number_of_systems=1, teacher_forcing_perc=0.2, reduce_ratio=0.5, 
                 dataset_length=40000, augment=False, tokenization_mode="bekern") -> None:
        super().__init__(teacher_forcing_perc, augment)
        tokenization_mode = validate_tokenization_mode(tokenization_mode)
        self.generator = _LazyVerovioGenerator(
            sources="antoniorv6/grandstaff-ekern",
            split=split,
            tokenization_mode=tokenization_mode,
        )
        
        self.num_sys_gen = number_of_systems
        self.dataset_len = dataset_length
        self.reduce_ratio = reduce_ratio
        self.tokenization_mode = tokenization_mode

    def __getitem__(self, index):
        generator = self.generator.get()
        x, y = _retry_synthetic_sample(generator.generate_music_system_image)

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return self.dataset_len

class RealDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, teacher_forcing_perc=0.2, reduce_ratio=1.0, 
                augment=False, tokenization_mode="bekern") -> None:
       super().__init__(teacher_forcing_perc, augment)
       tokenization_mode = validate_tokenization_mode(tokenization_mode)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.real_source = _ArrowOMRSource(
           data_path,
           split,
           tokenization_mode,
           reduce_ratio,
       )
       
    def __getitem__(self, index):
       x, y = self.real_source[index]

       if self.augment:
           x = augment(x)
       else:
           x = convert_img_to_tensor(x)

       y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
       decoder_input = self.apply_teacher_forcing(y)
       return x, decoder_input, y

    def __len__(self):
       return len(self.real_source)

    def get_gt(self):
       return self.real_source.iter_token_sequences()

class CurriculumTrainingDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, 
                teacher_forcing_perc=0.2, 
                reduce_ratio=1.0,
                augment=False, 
                tokenization_mode="bekern",
                skip_steps: int = 0,
                step_counter: _SharedStepCounter | None = None,
                capture_input_metadata: bool = False) -> None:
       super().__init__(teacher_forcing_perc, augment)
       tokenization_mode = validate_tokenization_mode(tokenization_mode)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.real_source = _ArrowOMRSource(
           data_path,
           split,
           tokenization_mode,
           reduce_ratio,
       )
       self.generator = _LazyVerovioGenerator(
           sources="antoniorv6/grandstaff-ekern",
           split="train",
           tokenization_mode=tokenization_mode,
       )
       
       self.max_synth_prob = 0.9
       self.min_synth_prob = 0.2
       self.finetune_steps = 200000
       self.increase_steps = CL_CURRICULUM_STAGE_STEPS
       self.num_cl_steps = CL_SYNTHETIC_STAGES
       self.max_cl_steps = CL_REAL_DATA_START_STEP
       self.curriculum_stage_beginning = 2
       self.step_counter = step_counter or _SharedStepCounter(skip_steps)
       self.capture_input_metadata = bool(capture_input_metadata)
    
    def linear_scheduler_synthetic(self, step):
        return self.max_synth_prob + round((step - self.max_cl_steps) * (self.min_synth_prob - self.max_synth_prob) / self.finetune_steps, 4)

    def __getitem__(self, index):
        step = self.step_counter.reserve()
        stage = (step // self.increase_steps) + self.curriculum_stage_beginning
        gen_author_title = np.random.rand() > 0.5
        input_metadata = None
        
        if stage < (self.num_cl_steps + self.curriculum_stage_beginning):
           generator = self.generator.get()
           x, y = _retry_synthetic_sample(
               lambda: generator.generate_full_page_score(
                   max_systems=random.randint(1, stage),
                   strict_systems=True,
                   strict_height=False,
                   include_author=gen_author_title,
                   include_title=gen_author_title,
               )
           )
        else:
            probability = max(self.linear_scheduler_synthetic(step), self.min_synth_prob)
            if random.random() > probability:
                if getattr(self, "capture_input_metadata", False):
                    x, y, input_metadata = self.real_source.get_with_resize_metadata(index)
                else:
                    x, y = self.real_source[index]
            else:
                generator = self.generator.get()
                x, y = _retry_synthetic_sample(
                    lambda: generator.generate_full_page_score(
                        max_systems=random.randint(2, 4),
                        strict_systems=False,
                        strict_height=False,
                        include_author=gen_author_title,
                        include_title=gen_author_title,
                    )
                )

        if getattr(self, "capture_input_metadata", False) and input_metadata is None:
            raw_shape = list(np.asarray(x).shape)
            input_metadata = {
                "source": "synthetic",
                "raw_shape_hwc": raw_shape,
                "intermediate_shape_hwc": raw_shape,
            }

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
        decoder_input = self.apply_teacher_forcing(y)
        sample = (x, decoder_input, y)
        if input_metadata is not None:
            input_metadata["final_shape_nchw"] = list(x.shape)
            return (*sample, input_metadata)
        return sample

    def __len__(self):
       return len(self.real_source)

    def get_gt(self):
       return self.real_source.iter_token_sequences()


class SynthToRealDataset(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, 
                teacher_forcing_perc=0.2, 
                reduce_ratio=1.0,
                augment=False, 
                tokenization_mode="bekern",
                step_counter: _SharedStepCounter | None = None) -> None:
       super().__init__(teacher_forcing_perc, augment)
       tokenization_mode = validate_tokenization_mode(tokenization_mode)
       self.reduce_ratio = reduce_ratio
       self.tokenization_mode = tokenization_mode
       self.real_source = _ArrowOMRSource(
           data_path,
           split,
           tokenization_mode,
           reduce_ratio,
       )
       self.generator = _LazyVerovioGenerator(
           sources="antoniorv6/grandstaff-ekern",
           split="train",
           tokenization_mode=tokenization_mode,
       )
       
       self.synth_pretraining_steps = SR_REAL_DATA_START_STEP
       self.step_counter = step_counter or _SharedStepCounter()

    def __getitem__(self, index):
        step = self.step_counter.reserve()
        gen_author_title = np.random.rand() > 0.5
        if step < self.synth_pretraining_steps:
            generator = self.generator.get()
            x, y = _retry_synthetic_sample(
                lambda: generator.generate_full_page_score(
                    max_systems=random.randint(2, 4),
                    strict_systems=False,
                    strict_height=False,
                    include_author=gen_author_title,
                    include_title=gen_author_title,
                )
            )
        else:
            x, y = self.real_source[index]

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y if token != '']))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def __len__(self):
       return len(self.real_source)

    def get_gt(self):
       return self.real_source.iter_token_sequences()

# CL1 for SMT
class SyntheticGrandStaffDataset(LightningDataModule):
    encoder_unfreeze_step = None
    curriculum_step_offset = 0

    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode

        self.train_dataset: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="train", dataset_length=40000, augment=True, tokenization_mode=self.tokenization_mode)
        self.val_dataset: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="val", dataset_length=1000, augment=False, tokenization_mode=self.tokenization_mode)
        self.test_dataset: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="test", dataset_length=1000, augment=False, tokenization_mode=self.tokenization_mode)
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#

        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)

    def get_max_height(self) -> int:
        return 2512

    def get_max_width(self) -> int:
        return 2512

    def get_max_length(self) -> int:
        return 4360

    def train_dataloader(self):
        return _build_dataloader(
            self.train_dataset,
            self.batch_size,
            self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return _build_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self):
        return _build_dataloader(self.test_dataset, self.batch_size, self.num_workers)

# CL2 and CL3
def _restored_samples_seen(trainer) -> int:
    if trainer is None:
        return 0
    lightning_module = getattr(trainer, "lightning_module", None)
    if lightning_module is None or not hasattr(lightning_module, "samples_seen"):
        raise RuntimeError("Trainer module is missing the restored samples_seen counter")
    samples_seen = lightning_module.samples_seen
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int) or samples_seen < 0:
        raise ValueError(f"samples_seen must be a non-negative integer, got {samples_seen!r}")
    return samples_seen


class CLFinetuningDataset(LightningDataModule):
    encoder_unfreeze_step = CL_REAL_DATA_START_STEP

    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode
        self.skip_steps: int = config.data.skip_steps
        self.step_counter = _SharedStepCounter(self.skip_steps)
        self.train_dataset = CurriculumTrainingDataset(data_path=self.data_path, split="train", 
                                                       augment=True, 
                                                       tokenization_mode=self.tokenization_mode,
                                                       reduce_ratio=config.data.reduce_ratio,
                                                       skip_steps=self.skip_steps,
                                                       step_counter=self.step_counter,
                                                       capture_input_metadata=True)
        self.val_dataset = RealDataset(data_path=self.data_path, split="val", augment=False, 
                                       tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.test_dataset = RealDataset(data_path=self.data_path, split="test", augment=False, 
                                        tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)

    @property
    def curriculum_step_offset(self) -> int:
        return self.skip_steps
        
    def train_dataloader(self):
        samples_seen = _restored_samples_seen(self.trainer)
        self.step_counter.reset(samples_seen + self.skip_steps)
        return _build_dataloader(
            self.train_dataset,
            self.batch_size,
            self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return _build_dataloader(self.val_dataset, self.batch_size, self.num_workers)
    
    def test_dataloader(self):
        return _build_dataloader(self.test_dataset, self.batch_size, self.num_workers)

class SynthRealFinetuningDataset(LightningDataModule):
    encoder_unfreeze_step = SR_REAL_DATA_START_STEP
    curriculum_step_offset = 0

    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode
        self.step_counter = _SharedStepCounter()
        self.train_dataset = SynthToRealDataset(data_path=self.data_path, split="train", 
                                                       augment=True, 
                                                       tokenization_mode=self.tokenization_mode,
                                                       reduce_ratio=config.data.reduce_ratio,
                                                       step_counter=self.step_counter)
        self.val_dataset = RealDataset(data_path=self.data_path, split="val", augment=False, 
                                       tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        self.test_dataset = RealDataset(data_path=self.data_path, split="test", augment=False, 
                                        tokenization_mode=self.tokenization_mode, reduce_ratio=config.data.reduce_ratio)
        
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        samples_seen = _restored_samples_seen(self.trainer)
        self.step_counter.reset(samples_seen)
        return _build_dataloader(
            self.train_dataset,
            self.batch_size,
            self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return _build_dataloader(self.val_dataset, self.batch_size, self.num_workers)
    
    def test_dataloader(self):
        return _build_dataloader(self.test_dataset, self.batch_size, self.num_workers)
