"""Preset MuSViT-variant lists for the embeddings sweep.

Only MuSViT (LSMT-MAE) encoders are shipped in this repository; the former
non-MuSViT baselines have been removed. ``full`` also includes the local
``models/MAE-8X8-Small`` checkpoint, which must be provided by the user at
``experiments/embeddings_test/models/MAE-8X8-Small`` (it is not downloaded).
"""

# The three published MuSViT (LSMT-MAE) variants.
MUSVIT_MODELS = [
    "carlospm12/LSMT-MAE-Small-1024-16",
    "carlospm12/LSMT-MAE-Base-1024-16",
    "carlospm12/LSMT-MAE-Large-1024-16",
]

# One quick variant, handy for smoke tests.
LIGHT_MODELS = ["carlospm12/LSMT-MAE-Small-1024-16"]

# The three published MuSViT variants.
DEFAULT_MODELS = list(MUSVIT_MODELS)

# Everything, including the local MAE checkpoint (must exist on disk).
FULL_MODELS = [*MUSVIT_MODELS, "models/MAE-8X8-Small"]

MODEL_SETS = {
    "light": LIGHT_MODELS,
    "default": DEFAULT_MODELS,
    "full": FULL_MODELS,
}
