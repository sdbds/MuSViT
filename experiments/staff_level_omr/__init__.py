"""Staff-level Optical Music Recognition experiment (MuSViT + BiLSTM/CTC)."""

import os


# Training and tests must not contact PyPI merely because Albumentations imports.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
