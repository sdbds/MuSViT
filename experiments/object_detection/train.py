"""Training driver for MuSViT-backbone Faster R-CNN object detection.

Builds a Faster R-CNN whose backbone is a MuSViT ViT (fully fine-tuned, frozen,
or LoRA-adapted) and trains it on a COCO-format detection dataset. The public
entry point is :func:`train`; the ``__main__`` guard lets the module be run
directly, but the canonical way is ``musvit object-detection``.

Requires a CUDA device for anything beyond a smoke test.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CocoDetectionDataset, collate_fn
from .models.detector_factory import build_detector
from .models.token_grid import set_model_fixed_size_and_resize_pe


def _apply_finetuning(model, mode):
    """Configure which backbone parameters train (mutates ``model`` in place).

    - ``full``   : train the whole MuSViT backbone.
    - ``frozen`` : freeze the ViT; train only FPN + detection heads.
    - ``lora``   : freeze the ViT and inject LoRA adapters on the attention
                   query/key/value projections (adapters + heads train).
    """
    vit = model.backbone.core.vit
    if mode == "full":
        return
    if mode not in ("frozen", "lora"):
        raise ValueError(f"Unknown finetuning mode: {mode!r} (expected full|frozen|lora)")

    for p in vit.parameters():
        p.requires_grad = False

    if mode == "lora":
        from peft import LoraConfig, LoraModel

        cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
            target_modules=["query", "key", "value"], use_rslora=True,
        )
        # Wrapping keeps the module path under ``backbone.core.vit`` so the
        # optimizer's backbone param-group still captures the LoRA adapters.
        model.backbone.core.vit = LoraModel(vit, cfg, adapter_name="default")


def _build_optimizer(model, defaults, lr_backbone, lr_head, weight_decay):
    """AdamW with two param groups: MuSViT backbone (low LR) and everything else."""
    prefix = defaults["backbone_prefix"]  # "backbone.core.vit"
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith(prefix) else head_params).append(p)

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def _validation_loss(model, loader, device):
    """Mean total loss over the val set.

    Faster R-CNN only returns the loss dict in ``train`` mode, so we switch the
    model to train mode (under ``no_grad``) purely to read the losses; this is a
    cheap proxy for validation quality (full mAP would need ``pycocotools``).
    """
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)
        total += float(sum(losses.values()))
        n += 1
    return total / max(n, 1)


def train(args):
    """Fine-tune a MuSViT-backbone Faster R-CNN. ``args`` mirrors the CLI flags."""
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_ds = CocoDetectionDataset(args.train_images, args.train_ann)
    num_classes = train_ds.num_classes
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    val_dl = None
    if args.val_ann:
        val_ds = CocoDetectionDataset(args.val_images or args.train_images, args.val_ann)
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
        )

    print(f"[object-detection] {num_classes - 1} foreground classes (+background), "
          f"{len(train_ds)} training images, finetuning={args.finetuning}")

    model, defaults = build_detector(
        args.model, num_classes=num_classes,
        vit_small_model_path=args.vit_small_model_path,
        vit_base_model_path=args.vit_base_model_path,
        hf_token=args.hf_token, out_channels=args.out_channels,
        fixed_size=args.fixed_size,
    )
    # Keep the detector's input size aligned with MuSViT's fixed token grid.
    set_model_fixed_size_and_resize_pe(model, args.model, target_tokens=args.target_tokens)
    _apply_finetuning(model, args.finetuning)
    model.to(device)

    optimizer = _build_optimizer(
        model, defaults, args.lr_backbone, args.lr_head, args.weight_decay,
    )

    start_epoch = 0
    if args.from_checkpoint:
        ckpt = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[object-detection] resumed from {args.from_checkpoint} @ epoch {start_epoch}")

    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project="musvit-object-detection",
            name=args.experiment_name or None,
            config=vars(args),
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for images, targets in train_dl:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss)
            n += 1

        train_loss = running / max(n, 1)
        log = {"epoch": epoch, "train_loss": train_loss}
        msg = (f"Epoch [{epoch}/{args.epochs}] train_loss {train_loss:.4f} "
               f"({(time.time() - t0) / 60:.1f} min)")

        if val_dl is not None:
            val_loss = _validation_loss(model, val_dl, device)
            log["val_loss"] = val_loss
            msg += f" | val_loss {val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "args": vars(args)},
                    out_dir / "ckpt-best.pt",
                )

        print(msg)
        if run is not None:
            run.log(log)

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch, "args": vars(args)},
            out_dir / "ckpt-latest.pt",
        )

    if run is not None:
        run.finish()


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Train a MuSViT-backbone Faster R-CNN for object detection.")
    p.add_argument("--model", default="musvit_base", choices=["musvit_small", "musvit_base"])
    p.add_argument("--train_images", required=True, help="Directory with training images.")
    p.add_argument("--train_ann", required=True, help="COCO-format training annotations JSON.")
    p.add_argument("--val_images", default=None)
    p.add_argument("--val_ann", default=None, help="COCO-format validation annotations JSON.")
    p.add_argument("--vit_base_model_path", default="PRAIG/musvit")
    p.add_argument("--vit_small_model_path", default="PRAIG/musvit-light")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--finetuning", default="lora", choices=["full", "frozen", "lora"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--fixed_size", type=int, default=1024)
    p.add_argument("--target_tokens", type=int, default=4096)
    p.add_argument("--out_channels", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="weights")
    p.add_argument("--from_checkpoint", default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--device", type=int, default=0)
    return p


if __name__ == "__main__":
    train(_build_arg_parser().parse_args())
