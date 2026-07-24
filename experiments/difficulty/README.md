# MusViT for score difficulty estimation

Estimates the difficulty of a music score from its rendered pages, using
[MusViT](https://huggingface.co/PRAIG/musvit) as a **frozen** encoder.

The pipeline runs in three stages, each writing its output to disk:

```
PDFs ──prepare_images──> page PNGs ──get_embeddings──> one 768-d vector per page ──run_model──> difficulty class
                                        (frozen MusViT)                          (RNN / Transformer / MLP)
```

MusViT never trains: it turns each page into an embedding once, and only the classification head on
top of the page sequence is trained. A score's label is the mode of its page labels.

## Datasets

**The datasets are not distributed with this code.** The `cipi`, `fs` and `ps` collections must be
requested from the authors of the original study.

Once you have them, place each one as follows — the code reads nothing else:

```
PDFdifficulty/<dataset>/pdf/<sample_id>.pdf
PDFdifficulty/<dataset>/splits.json
```

`splits.json` maps each fold to its splits, and each split to `{sample_id: label}`, where `sample_id`
matches the PDF filename and `label` is a 0-based integer difficulty:

```json
{
  "0": {
    "train": {"score_001": 3, "score_002": 5},
    "val":   {"score_003": 2},
    "test":  {"score_004": 7}
  },
  "1": { "...": {} }
}
```

## Setup

```bash
uv sync
```

MusViT is public, but its repositories are gated behind accepting their conditions. Accept them on
[PRAIG/musvit](https://huggingface.co/PRAIG/musvit) and
[PRAIG/musvit-light](https://huggingface.co/PRAIG/musvit-light), then authenticate once:

```bash
huggingface-cli login
```

This is the only authentication step. **No token is ever passed as an argument or stored in this
repository** — `transformers` picks up your Hugging Face login (or an `HF_TOKEN` in the environment)
on its own.

To log to Weights & Biases, add `WANDB_API_KEY` to the repository-root `.env` (see the root
[Environment Variables](../../README.md#environment-variables) section); the `musvit` launcher loads
it automatically. Otherwise pass `--no_log` and everything runs offline.

Stage 1 needs [poppler](https://poppler.freedesktop.org/) (`brew install poppler`,
`apt install poppler-utils`). Stage 2 wants a GPU.

## Running an experiment

```bash
# 1. PDFs -> one PNG per page, named {sample_id}_{page}_{label}.png
uv run musvit difficulty prepare-images --dataset_name fs

# 2. Frozen MusViT -> one .npy embedding per page
uv run musvit difficulty embeddings --model_name PRAIG/musvit --dataset_name fs --architecture rnn

# 3. Train and test the head on one fold
uv run musvit difficulty run --model_name PRAIG/musvit --dataset_name fs \
    --architecture rnn --fold_idx 0 --epochs 100
```

`--model_name` accepts `PRAIG/musvit` or `PRAIG/musvit-light`; `--architecture` accepts `rnn`,
`transformer` or `mlp`. Add `--test` to skip training and evaluate an existing `ckpt-best.pt`.

Embeddings are cached per architecture, so run step 2 once for each head you intend to train
(`--architecture rnn`, then `transformer`, then `mlp`).

### On a cluster

`queue.sh` submits every (architecture x model x dataset) combination as a chain of 5 folds via
SLURM. Adapt the `#SBATCH` directives and the conda activation block in `launch_experiment.slurm` to
your site, then:

```bash
./queue.sh
```

## Outputs

| Path | Contents |
| --- | --- |
| `PDFdifficulty/<dataset>/images_per_score/` | Rendered pages, 300 dpi |
| `difficulty_embeddings/<arch>/<dataset>/<model>/` | One `.npy` per page, mean-pooled, 768-d |
| `weights/<...>/<fold>/ckpt-best.pt` | Best checkpoint by validation MSE |
| `output/<...>/<dataset>/<fold>.{pdf,json}` | Confusion matrix and per-score predictions |

