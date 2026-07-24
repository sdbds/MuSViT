import argparse
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from tqdm import tqdm

from ..musvitdifficulty.globals import DATASET_CHOICES


def process_pdf(sample, label, pdf_path, img_path, dpi):
    pdf_file = (pdf_path / sample).with_suffix(".pdf")

    if not pdf_file.exists():
        return f"PDF file {pdf_file} does not exist."

    try:
        images = convert_from_path(pdf_file, dpi=dpi)
    except (PDFPageCountError, ValueError) as e:
        return f"Failed to process {pdf_file}: {e}"
    except Exception as e:
        return f"Unexpected error with {pdf_file}: {e}"

    for i, img in enumerate(images):
        img.save(img_path / f"{sample}_{i}_{label}.png")

    return f"Processed {pdf_file}"


def main(args):
    dataset_path = Path("PDFdifficulty") / args.dataset_name

    with open(dataset_path / "splits.json", "r") as f:
        splits = json.load(f)

    split = splits[args.fold_idx]

    pdf_path = dataset_path / "pdf"
    img_path = dataset_path / "images_per_score"
    img_path.mkdir(parents=True, exist_ok=True)

    # The three splits of a fold partition the whole dataset, so this rasterizes every score.
    samples = []
    for subset in ["train", "val", "test"]:
        for sample, label in split[subset].items():
            samples.append((sample, label))

    print(f"Rasterizing {len(samples)} scores from {pdf_path} at {args.dpi} dpi...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_pdf, sample, label, pdf_path, img_path, args.dpi)
            for sample, label in samples
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Rasterize score PDFs into per-page PNGs.")
    argparser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Name of the dataset to rasterize.",
    )
    argparser.add_argument(
        "--fold_idx",
        type=str,
        required=False,
        default="0",
        help="Fold whose splits are read. Every fold covers the whole dataset.",
    )
    argparser.add_argument(
        "--dpi",
        type=int,
        required=False,
        default=300,
        help="Rasterization resolution.",
    )
    argparser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=8,
        help="Number of PDFs to convert in parallel.",
    )

    args = argparser.parse_args()

    main(args)
