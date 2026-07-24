<p align="center">
  <a  href="#"> <img src="https://raw.githubusercontent.com/OMR-PRAIG-UA-ES/MuSViT/refs/heads/main/resources/musvit_logo.svg" alt="MuSViT-logo" width="25%"></a>
</p>
<h1 align="center">MuSViT: A Foundation Vision Model for Sheet Music Representation</h1>
<h4 align="center">📄 Official publication at ECCV 2026.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.x-orange" alt="Python">
  <img src="https://img.shields.io/badge/uv-managed-6340ac?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black" alt="W&B">
  <img src="https://img.shields.io/static/v1?label=License&message=CC%20BY-NC-SA%204.0&color=blue" alt="License">
  <a href="https://arxiv.org/abs/2606.31811"><img src="https://img.shields.io/badge/arXiv-2606.31811-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/PRAIG/musvit"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-PRAIG%2Fmusvit-yellow" alt="HuggingFace"></a>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#repository-layout">Repository Layout</a> •
  <a href="#installation">Installation</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#running-experiments">Running Experiments</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#license">License</a>
</p>

<a name=about></a>
## 📚 About

This repository contains the official code for **MuSViT: A Foundation Vision Model for Sheet Music Representation**, presented at ECCV 2026.

This is a **monorepo** collecting the experiments around MuSViT. Every experiment lives under [`experiments/`](experiments/) and is launched from the repository root through a single cross-platform CLI: **`musvit`**.

This project includes:
- A central CLI (`musvit`) to discover and run all experiments from the repository root.
- A full-page Optical Music Recognition (OMR) fine-tuning pipeline.
- A staff-level OMR fine-tuning pipeline (linear probing or LoRA).
- An object-detection fine-tuning pipeline (Faster R-CNN with a MuSViT backbone).
- An embeddings analysis pipeline correlating MuSViT embedding distances with transcription distances (single encoder or cross-platform sweep).
- A score-difficulty estimation pipeline built on frozen MuSViT page embeddings.

<a name=repository-layout></a>
## ⚙️ Repository Layout

| Folder | Description |
| --- | --- |
| [`musvit/`](musvit/) | Central launcher (`musvit` CLI): discovers and runs the experiments. |
| [`experiments/full_page_omr/`](experiments/full_page_omr/) | Fine-tune MuSViT for full-page Optical Music Recognition (OMR). |
| [`experiments/staff_level_omr/`](experiments/staff_level_omr/) | Fine-tune MuSViT for staff-level OMR (BiLSTM/CTC head; linear probing or LoRA). |
| [`experiments/object_detection/`](experiments/object_detection/) | Fine-tune MuSViT + Faster R-CNN for object detection (COCO-format data). |
| [`experiments/embeddings_test/`](experiments/embeddings_test/) | Correlate MuSViT embedding distances with transcription distances. |
| [`experiments/difficulty/`](experiments/difficulty/) | Estimate score difficulty from frozen MuSViT page embeddings. |

<a name=installation></a>
## 🔧 Installation

The whole monorepo shares a single [uv](https://docs.astral.sh/uv/) project defined at the repository root ([`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock)). Install the environment once from the root:

```bash
uv sync
```

This also installs the `musvit` console command into the project environment.

### Windows Cairo dependency

Full-page OMR and its tests import CairoSVG, which requires a native Cairo DLL on Windows. Install Cairo, then expose the directory containing `libcairo-2.dll` through the environment before launching training or `pytest`:

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES = "C:\path\to\cairo\bin"
$env:PATH = "$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
uv run pytest -q
```

`2.full_page_omr.ps1` first checks its optional `Runtime.cairo_dll_directory`, then `CAIROCFFI_DLL_DIRECTORIES`, then `PATH`. The repository does not commit a machine-specific Cairo directory.
Two experiments additionally rely on **system libraries** that `uv` does not manage:

- **ImageMagick** — used by `wand` in the full-page-OMR synthetic data generator (`brew install imagemagick`, or `apt install libmagickwand-dev`).
- **poppler** — used by `pdf2image` in the score-difficulty rasterization stage (`brew install poppler`, or `apt install poppler-utils`).

<a name=environment-variables></a>
## 🔑 Environment Variables

Experiments authenticate against [Weights & Biases](https://wandb.ai) and the [Hugging Face Hub](https://huggingface.co). These credentials are shared across the whole monorepo through a single `.env` file at the repository root.

1. Copy the template:
```bash
   cp .env.example .env
```
2. Fill in your keys in `.env`:
```dotenv
   WANDB_API_KEY=...
   HUGGINGFACE_KEY=...
```

`.env` is git-ignored; only `.env.example` is committed. The `musvit` launcher loads this `.env` once at startup and automatically bridges `HUGGINGFACE_KEY` to `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, so every experiment authenticates from the same key regardless of which variable it reads.

<a name=running-experiments></a>
## 🚀 Running Experiments

All experiments are launched from the repository root with `musvit`. List what is available:

```bash
uv run musvit list
```

Each subcommand maps its options directly onto the experiment's Python entrypoint, and `--help` is generated automatically:

```bash
uv run musvit full-page-omr --help
uv run musvit staff-level-omr --help
uv run musvit object-detection --help
uv run musvit embeddings run --help
uv run musvit embeddings sweep --help
uv run musvit difficulty run --help
```

### Full-page OMR (fine-tuning)

```bash
uv run musvit full-page-omr \
  --config_path experiments/full_page_omr/config/Polish_Scores/finetuning.json \
  --experiment_name my_experiment \
  --finetuning CL
```

Checkpoints and W&B logs are written inside `experiments/full_page_omr/` (`weights/`, `wandb_logs/`).

### Staff-level OMR (fine-tuning)

```bash
uv run musvit staff-level-omr \
  --ds_name catedrales \
  --model_name musvit \
  --method lora \
  --shape_patches '[8,128]'
```

Fine-tunes a MuSViT backbone with a BiLSTM/CTC head on cropped staves, by either linear probing (`--method linear_prob`, frozen backbone) or LoRA (`--method lora`). Checkpoints (`<model>_<ds>_<method>_<cols>_ctc.pt`) are written inside `experiments/staff_level_omr/`. Requires a GPU and editing the dataset paths in [`config.py`](experiments/staff_level_omr/config.py) to point at your local data.

### Object Detection (Faster R-CNN fine-tuning)

```bash
uv run musvit object-detection \
  --model musvit_base \
  --train_images /path/to/images \
  --train_ann /path/to/train.json \
  --val_images /path/to/images \
  --val_ann /path/to/val.json \
  --finetuning lora
```

Fine-tunes a Faster R-CNN with a MuSViT + FPN backbone. Annotations are read in **COCO format** (`images` / `annotations` / `categories`); convert DeepScores (or any detection set) to COCO and point `--train_ann` / `--val_ann` at the JSON files — the number of classes is inferred from `categories`. `--finetuning` selects `full`, `frozen` (freeze the ViT) or `lora` (LoRA adapters on q/k/v). Checkpoints (`ckpt-best.pt`, `ckpt-latest.pt`) are written inside `--out_dir` (under `experiments/object_detection/`); `--use_wandb` enables W&B logging and `--from_checkpoint` resumes. Requires a GPU.

### Embeddings analysis (single encoder)

```bash
uv run musvit embeddings run --model carlospm12/LSMT-MAE-Base-1024-16 --device 0
```

Embeddings, results and logs are written inside `experiments/embeddings_test/` (`embeddings/`, `results/`).

### Embeddings sweep (several MuSViT variants)

Runs the analysis over a preset list of MuSViT variants, writing a per-model log and a `summary.tsv`:

```bash
uv run musvit embeddings sweep --models_set light
uv run musvit embeddings sweep --models_set default --device 0
uv run musvit embeddings sweep --only carlospm12/LSMT-MAE-Large-1024-16
```

Model sets `light`, `default`, `full` (defined in [`experiments/embeddings_test/model_sets.py`](experiments/embeddings_test/model_sets.py)) contain only MuSViT (LSMT-MAE) variants. `full` additionally expects a local `models/MAE-8X8-Small` checkpoint under `experiments/embeddings_test/`.

> ⚠️ **Note on GPU memory**  
> The sweep loads encoders one after another in the same process and frees GPU memory between models. If you hit out-of-memory errors with the largest sets, run the encoders individually with `musvit embeddings run`.

### Score difficulty (3-stage pipeline)

Difficulty estimation runs as three stages that share the same `--dataset_name` and `--architecture` (`rnn`, `transformer` or `mlp`):

```bash
# 1. Rasterize score PDFs into per-page images.
uv run musvit difficulty prepare-images --dataset_name fs

# 2. Extract frozen-MuSViT page embeddings.
uv run musvit difficulty embeddings --model_name PRAIG/musvit --dataset_name fs --architecture rnn

# 3. Train and test the difficulty classifier head.
uv run musvit difficulty run --model_name PRAIG/musvit --dataset_name fs --architecture rnn --fold_idx 0 --epochs 100
```

Artifacts (`PDFdifficulty/`, `difficulty_embeddings/`, `weights/`, `output/`) are written inside `experiments/difficulty/`. Stages 2-3 require a GPU; stage 1 needs the system `poppler` library (used by `pdf2image`). A SLURM launcher (`queue.sh`, `launch_experiment.slurm`) is provided for clusters.

<a name=citations></a>
## 📖 Citations

```bibtex
@inproceedings{penarrubia2026musvit,
  title     = {MuSViT: A Foundation Vision Model for Sheet Music Representation},
  author    = {Penarrubia, Carlos and Rios-Vila, Antonio and Fuentes-Martinez, Eliseo 
              and Martinez-Sevilla, Juan C. and Castellanos, Francisco J. and 
              Alfaro-Contreras, Maria and Calvo-Zaragoza, Jorge},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

<a name=acknowledgements></a>
## 🙏 Acknowledgements

The authors gratefully acknowledge Edward Guo, on behalf of IMSLP/Petrucci Music Library, for providing access to the data used to train the models.

This publication is part of the LEMUR project PID2023-148259NB-I00, funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU. The first author is supported by the University of Alicante through the FPU Program (UAFPU22-19). The third author is supported by a predoctoral contract associated with the LEMUR project. The fourth author is supported by a predoctoral contract from grant CISEJI/2023/9 "Programa para el apoyo a personas investigadoras con talento (Plan GenT) de la Generalitat Valenciana".

<a name=license></a>
## 📝 License

This work is under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

> ⚠️ **Disclaimer**  
> This github is under active development. Some parts of the codebase may not work as expected or could change without notice.  
> Please proceed with caution and check back for updates.
