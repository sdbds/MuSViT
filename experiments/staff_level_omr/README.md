# Staff-level Optical Music Recognition with MusViT

Fine-tune a pre-trained **MusViT** Vision Transformer for **staff-level Optical
Music Recognition (OMR)**: given the image of a single staff (a cropped score
region), the model transcribes it into a left-to-right sequence of music
symbols.

The backbone is a self-supervised ViT for music-score images published by the
[PRAIG](https://huggingface.co/PRAIG) group on the Hugging Face Hub. On top of
it we add a lightweight recurrent + CTC head and adapt the whole thing to each
target collection with either **linear probing** or **LoRA**.

---

## How it works

```
staff image
   │
   ▼
MusViT backbone (frozen, or LoRA-adapted)     ← pre-trained ViT patch encoder
   │   patch tokens
   ▼
reshape to a (rows × cols) grid, drop [CLS]
   │
   ▼
linear projection  →  256-d per patch
   │
   ▼
mean-pool over rows (collapse staff height)   ← one feature vector per column
   │   sequence of length = cols
   ▼
2-layer bidirectional LSTM
   │
   ▼
linear classifier  →  per-column class logits
   │
   ▼
CTC loss (training)  /  greedy CTC decode (inference)
   │
   ▼
symbol id sequence  →  Character/Symbol Error Rate (CER)
```

The key idea: the ViT turns the staff image into a grid of patch features; the
vertical (staff-height) axis is averaged away so each **column** becomes one
frame of a sequence, and **CTC** aligns that column sequence to the shorter
symbol sequence without needing explicit symbol positions.

### Two fine-tuning strategies

| Method         | Backbone        | What trains                          | Typical grid | Batch |
|----------------|-----------------|--------------------------------------|--------------|-------|
| `linear_prob`  | frozen          | projection + LSTM + classifier only  | `8 64`       | 16    |
| `lora`         | frozen + LoRA   | LoRA adapters (q/k/v) + head         | `8 128`      | 8     |

With `linear_prob`, inputs are padded to a fixed 64×64 patch grid and the model
slices out the real rows. With `lora`, the ViT interpolates its positional
embeddings to the exact `rows × cols` grid, so wider inputs are supported.

---

## Repository layout

```
.
├── arguments.py            # CLI argument definitions (hyper-parameters, switches)
├── augments.py             # albumentations augmentation pipeline
├── config.py               # dataset paths and pre-trained backbone definitions
├── datasets.py             # CTC_ds: PyTorch Dataset yielding (image, target, length)
├── model.py                # ViTRNN: ViT backbone + BiLSTM + CTC head
├── train.py                # training entry point (data → train → evaluate)
├── executions.sh           # ready-to-run experiment commands for every dataset
└── utils/
    ├── data_utils.py       # load pairs, encode/pad sequences, length filtering
    └── utils.py            # CER metric, test loop, backbone loader
```

---

## Requirements

The code runs inside the shared MuSViT monorepo environment (Python 3.11+) and
needs a CUDA-capable GPU (training calls `.cuda()` directly). Core dependencies:

- `torch`, `torchvision`
- `transformers`  (loads the MusViT checkpoint; requires `trust_remote_code`)
- `peft`          (LoRA adapters)
- `albumentations`, `opencv-python`  (augmentation)
- `scikit-learn`  (label encoding, train/test split)
- `editdistance`  (CER metric)
- `Pillow`, `numpy`

Install the shared monorepo environment once from the repository root (this pulls
in `torch`, `transformers`, `peft`, `albumentations`, … and the `musvit` command):

```bash
uv sync
```

---

## Data

Each dataset is a single folder of paired files:

```
<id>_region.png   # the cropped staff-region image
<id>_gt.txt       # ground truth: whitespace-separated music symbols
```

`load_image_gt_pairs` discovers these pairs automatically and warns about any
image missing its ground truth.

The datasets referenced here (`capitan`, `catedrales`, `fmt`, `guatemala`,
`seils`) are distributed **on request or are private** — obtain them from their
respective authors. Once you have them, edit `config.py` and point each entry in
`data_paths` at your local folder:

```python
data_paths = {
    'capitan':    '/your/path/capitan/data',
    'catedrales': '/your/path/catedrales/data',
    ...
}
```

---

## Pre-trained models

`config.py` defines the available backbones (pulled from the Hugging Face Hub on
first use):

| key            | Hub repo             | patch size | hidden dim |
|----------------|----------------------|------------|------------|
| `musvit`       | `PRAIG/musvit`       | 16         | 768        |
| `musvit_light` | `PRAIG/musvit-light` | 16         | 384        |

No manual download is needed; `transformers` fetches and caches the weights.

---

## Usage

### Single training run

```bash
uv run musvit staff-level-omr \
    --model_name=musvit \
    --ds_name=catedrales \
    --method=lora \
    --batch_size=8 \
    --start_eval 20 \
    --shape_patches '[8,128]' \
    --lr=0.0003
```

### All experiments

`executions.sh` runs every dataset for both methods (linear probing then LoRA):

```bash
bash experiments/staff_level_omr/executions.sh
```

### Arguments

| Argument          | Default        | Description                                                            |
|-------------------|----------------|------------------------------------------------------------------------|
| `--ds_name`       | `catedrales`   | Dataset key from `config.data_paths`.                                  |
| `--model_name`    | `musvit`       | Backbone key from `config.data_models` (`musvit` or `musvit_light`).   |
| `--method`        | `lora`         | `linear_prob` or `lora`.                                               |
| `--shape_patches` | `8 64`         | Patch grid `rows cols`; `cols` is the CTC time-axis length.           |
| `--batch_size`    | `8`            | Mini-batch size for all dataloaders.                                   |
| `--start_eval`    | `20`           | First epoch at which validation / checkpointing begins.               |
| `--lr`            | `0.0003`       | Adam learning rate.                                                    |

> Make sure the `--ds_name` you pass exists in `config.data_paths` and points at
> local data; `--model_name` must be `musvit` or `musvit_light`.

---

## Training details

- **Splits.** 80% train / 10% validation / 10% test, with a fixed seed for
  reproducibility.
- **Vocabulary.** Symbols are integer-encoded with a `LabelEncoder`; all ids are
  shifted up by 1 so that id `0` is reserved for the **CTC blank**.
- **Length filtering.** CTC needs input length ≥ target length, so training
  samples whose symbol sequence is longer than `cols` are dropped.
- **Loss.** `F.ctc_loss` with `zero_infinity=True`.
- **Checkpointing & early stopping.** Validation CER is tracked from
  `--start_eval` onward; the best model is saved, and training stops after 30
  evaluations without improvement.
- **Reproducibility.** Main-process RNGs are seeded, and each DataLoader worker
  is re-seeded per epoch (including the albumentations RNG) via
  `_worker_init_fn`.

### Output checkpoint

The best model is written to the working directory as:

```
<model_name>_<ds_name>_<method>_<cols>_ctc.pt
# e.g.  musvit_catedrales_lora_128_ctc.pt
```

After training, the script reloads this checkpoint and prints **train** and
**test** CER.

---

## Evaluation metric

Quality is measured with **CER (Character/Symbol Error Rate)** — the total edit
distance between predicted and reference symbol sequences, normalised by total
reference length. Lower is better (`0.0` = perfect). `engine_test` also prints
one ground-truth / prediction pair per evaluation as a quick sanity check.

---

```bash
python augments.py   # writes aux0.png ... aux9.png
```

---
