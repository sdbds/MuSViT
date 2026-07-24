import os

# huggingface_hub reads this at import time, so it has to be set before transformers is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transformers import ViTModel

from .globals import MODEL_CHOICES

EMBEDDING_FOLDER = "difficulty_embeddings"

# MusViT is a ViT-MAE with 16x16 patches over a 1024x1024 page, so last_hidden_state
# is (B, 1 + (1024 / 16) ** 2, 768) = (B, 4097, 768): one CLS token plus 4096 patches.
IMAGE_SIZE = 1024
PATCH_SIZE = 16
EXPECTED_TOKENS = 1 + (IMAGE_SIZE // PATCH_SIZE) ** 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_img = T.Compose(
    [
        T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        T.ToTensor(),
    ]
)


def load_musvit(model_name):
    if model_name not in MODEL_CHOICES:
        raise ValueError(
            f"Model {model_name} is not a MusViT checkpoint. Choose one of {MODEL_CHOICES}."
        )

    # The MusViT repositories are public but gated ("accept the conditions"), so authentication is
    # ambient: `huggingface-cli login`, or an HF_TOKEN in the environment. No token is passed here.
    # AutoModel would return masked and shuffled patches; the model card asks for ViTModel.
    # The checkpoint carries no pooler, and we read last_hidden_state, so skip it rather than
    # let ViTModel initialize one at random.
    model = ViTModel.from_pretrained(
        model_name,
        add_pooling_layer=False,
        trust_remote_code=True,
    )

    return model.eval().to(device)


@torch.no_grad()
def get_embeddings(model_name, files, dataset_name, architecture_name):
    model = load_musvit(model_name)
    print(f"Using model: {model_name} on {device}")

    output_path = Path(EMBEDDING_FOLDER) / architecture_name / dataset_name / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(files):
        image = Image.open(sample).convert("RGB")
        pixel_values = transform_img(image).unsqueeze(0).to(device)

        out = model(pixel_values=pixel_values).last_hidden_state

        if out.shape[1] != EXPECTED_TOKENS:
            raise RuntimeError(
                f"Expected {EXPECTED_TOKENS} tokens at {IMAGE_SIZE}px, got {tuple(out.shape)}."
            )

        # Mean pooling over the token axis. The classifier heads consume a single vector per page,
        # so pooling here rather than at load time stores ~3 kB per page instead of ~12 MB.
        embedding = out[0].mean(dim=0)

        np.save(output_path / f"{sample.stem}.npy", embedding.cpu().numpy())
