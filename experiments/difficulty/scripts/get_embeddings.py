import argparse

from pathlib import Path

from ..musvitdifficulty.encoder import get_embeddings
from ..musvitdifficulty.globals import MODEL_CHOICES, DATASET_CHOICES, ARCHITECTURE_CHOICES

def main(args):
    base_path = Path("PDFdifficulty") / args.dataset_name / "images_per_score"
    files = list(base_path.glob("*.png"))

    if not files:
        raise FileNotFoundError(
            f"No page images in {base_path}. Run `musvit difficulty prepare-images "
            f"--dataset_name {args.dataset_name}` first."
        )

    print(f"Number of files found: {len(files)}")
    print(f"Model name: {args.model_name}")
    print(f"Dataset name: {args.dataset_name}")
    print("Starting to get embeddings...")

    get_embeddings(
        model_name=args.model_name,
        files=files,
        dataset_name=args.dataset_name,
        architecture_name=args.architecture_name,
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Get MusViT embeddings for a dataset.")
    argparser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_CHOICES,
        help="Name of the model to use for embeddings.",
        required=True,
    )
    argparser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Name of the dataset to use for embeddings.",
    )
    argparser.add_argument(
        "--architecture_name",
        type=str,
        required=True,
        choices=ARCHITECTURE_CHOICES,
    )

    args = argparser.parse_args()

    main(args)
