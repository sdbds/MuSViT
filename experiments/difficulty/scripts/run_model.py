import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb
from ..musvitdifficulty.dataset import PDFDifficultyDataset
from ..musvitdifficulty.rnn import RNNClassifier
from ..musvitdifficulty.transformer import TransformerClassifier
from ..musvitdifficulty.mlp import MLPClassifier
from ..musvitdifficulty.globals import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn_rnn(batch):
    sequences, pages, labels = zip(*batch)
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(pages), torch.tensor(labels)

def collate_fn_transformer(batch):
    sequences, pages, labels = zip(*batch)
    max_len = max(pages)
    padded_sequences = [torch.cat([sequence, torch.zeros(max_len - sequence.size(0), sequence.size(1))]) for sequence in sequences]
    return torch.stack(padded_sequences), torch.tensor(pages), torch.tensor(labels)

def get_dataloaders(dataset_path, embeddings_path, collate_fn, batch_size=8, fold_idx=0):
    with open(dataset_path / 'splits.json', 'r') as f:
        splits = json.load(f)
    split = splits[fold_idx]

    train_split = set(split['train'])
    val_split = set(split['val'])
    test_split = set(split['test'])

    train_dataset = PDFDifficultyDataset(embeddings_path, train_split)
    val_dataset = PDFDifficultyDataset(embeddings_path, val_split)
    test_dataset = PDFDifficultyDataset(embeddings_path, test_split)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

    embedding_dim = train_dataset[0][0].shape[-1]
    num_classes = train_dataset.get_num_classes()

    return train_loader, val_loader, test_loader, embedding_dim, num_classes

def get_model(architecture_type, embedding_dim, hidden_dim, num_classes):
    if architecture_type == "rnn":
        return RNNClassifier(embedding_dim = embedding_dim, hidden_dim = hidden_dim, num_classes = num_classes)
    elif architecture_type == "transformer":
        return TransformerClassifier(embedding_dim = embedding_dim, hidden_dim = hidden_dim, num_classes = num_classes)
    else:
        return MLPClassifier(embedding_dim = embedding_dim, hidden_dim = hidden_dim, num_classes = num_classes)

def train(model, train_loader, val_loader, lr, epochs, weights_path, no_log):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()

    best_mse = float("inf")
    patience = 20
    counter = 0
    improvement = 0.01

    for epoch in range(1, epochs + 1):
        train_loss = total = 0

        model.train()
        for train_x, train_pages, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            out = model(train_x, train_pages)

            if not torch.isfinite(out).all():
                print("NaNs detected!")
                print("out shape:", out.shape)
                print("min/max:", out.min().item(), out.max().item())
                raise RuntimeError("NaNs in model output")

            loss = loss_fn(out, train_y)
            train_loss += loss.item() * train_y.size(0)
            total += train_y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= total

        val_loss, val_acc_0, val_acc_1, val_mse, val_dist = evaluate(model, val_loader, loss_fn)

        if not no_log:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc_0": val_acc_0,
                "val_acc_1": val_acc_1,
                "val_mse": val_mse,
                "val_dist": val_dist,
            })

        print(f"Epoch {epoch} | training loss: {train_loss:.4f} | validation loss: {val_loss:.4f} | val acc 0: {val_acc_0:.4f}% | val acc 1: {val_acc_1:.4f}% | val mse: {val_mse:.2f} | val dist: {val_dist:.2f}")

        if val_mse < best_mse * (1 - improvement):
            best_mse = val_mse
            counter = 0

            torch.save(model.state_dict(), weights_path / "ckpt-best.pt")
            torch.save(model.state_dict(), weights_path / f"ckpt-e{epoch}-{val_acc_0:.4f}.pt")
        else:
            counter += 1

        torch.save(model.state_dict(), weights_path / "ckpt-latest.pt")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

def evaluate(model, loader, loss_fn=None, args=None):
    if loss_fn:
        loss = 0
    correct_0 = correct_1 = mse = dist = total = 0

    cm = args is not None
    if cm:
        preds = []
        ys = []

    model.eval()
    with torch.no_grad():
        for x, pages, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, pages)

            if loss_fn:
                i_loss = loss_fn(out, y)
                loss += i_loss.item() * y.size(0)

            total += y.size(0)
            pred = out.argmax(dim = 1)

            correct_0 += (pred == y).sum().item()
            correct_1 += (torch.abs(pred - y) <= 1).sum().item()
            mse += F.mse_loss(pred.float(), y.float(), reduction='sum').item()
            dist += F.l1_loss(pred.float(), y.float(), reduction='sum').item()

            if cm:
                preds.append(pred.cpu())
                ys.append(y.cpu())

    if loss_fn:
        loss /= total
    acc_0 = (correct_0 / total) * 100
    acc_1 = (correct_1 / total) * 100
    mse /= total
    dist /= total

    if cm:
        preds_tensor = torch.cat(preds)
        ys_tensor = torch.cat(ys)
        plot_confusion_matrix(preds_tensor, ys_tensor, args)

    return (loss, acc_0, acc_1, mse, dist) if loss_fn else (acc_0, acc_1, mse, dist)

def plot_confusion_matrix(predictions, targets, args):
    cmaps = {
        'cipi': plt.cm.Blues,
        'fs': plt.cm.Greens,
        'ps': plt.cm.Reds
    }

    cm = confusion_matrix(targets, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(values_format=".1f", cmap=cmaps.get(args.dataset_name, plt.cm.Greys), ax=ax)
    plt.title("Confusion Matrix (%)")

    output_path = Path("output") / f"{args.architecture_type}_{args.model_name.replace('/', '-')}" / args.dataset_name
    output_path.mkdir(parents = True, exist_ok = True)

    plt.savefig(output_path / f"{args.fold_idx}.pdf", bbox_inches='tight')

    if not args.no_log:
        wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    df = pd.DataFrame({
        "id": [i for i in range(len(predictions))],
        "prediction": predictions,
        "target": targets
    })

    df.to_json(output_path / f"{args.fold_idx}.json", orient='records', indent=2)

def main(args):
    dataset_path = Path("PDFdifficulty") / args.dataset_name
    embeddings_path = Path("difficulty_embeddings") / args.architecture_type / args.dataset_name / args.model_name

    # TODO: Make these hyperparameters more flexible (passed as args?)
    batch_size = 8
    hidden_dim = 256
    lr = 1e-3

    weights_path = Path("weights") / f"{args.architecture_type}_{args.model_name.replace('/', '-')}_{args.dataset_name}_b{batch_size}_e{args.epochs}_d{hidden_dim}_lr{lr}" / f"{args.fold_idx}"
    weights_path.mkdir(parents = True, exist_ok = True)

    if not args.no_log:
        wandb.init(
            project = "eswa-score-difficulty",
            group = f"{args.timestamp}_eswa_{args.architecture_type}_{args.model_name}_{args.dataset_name}_b{batch_size}_e{args.epochs}_d{hidden_dim}_lr{lr}",
            name = f"fold_{args.fold_idx}",
            config = {
                "architecture": args.architecture_type,
                "dataset": args.dataset_name,
                "model": args.model_name,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": args.epochs,
                "hidden_dim": hidden_dim,
                "fold": args.fold_idx,
            }
        )

    collate_fn = collate_fn_transformer if args.architecture_type == "transformer" else collate_fn_rnn
    train_loader, val_loader, test_loader, embedding_dim, num_classes = get_dataloaders(dataset_path, embeddings_path, collate_fn, batch_size, args.fold_idx)

    model = get_model(args.architecture_type, embedding_dim, hidden_dim, num_classes)
    model = model.to(device)

    if not args.test:
        train(model, train_loader, val_loader, lr, args.epochs, weights_path, args.no_log)

    checkpoint = torch.load(weights_path / "ckpt-best.pt", map_location=device)
    model.load_state_dict(checkpoint)
    test_acc_0, test_acc_1, test_mse, test_dist = evaluate(model, test_loader, args=args)

    if not args.no_log:
        wandb.log({
            "test_acc_0": test_acc_0,
            "test_acc_1": test_acc_1,
            "test_mse": test_mse,
            "test_dist": test_dist
        })

    print(f"Test | acc 0: {test_acc_0:.4f}% | acc 1: {test_acc_1:.4f}% | mse: {test_mse:.2f}% | dist: {test_dist:.2f}")

    if not args.no_log:
        wandb.finish()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train and test model for classification.")
    argparser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
        help="Name of the model used for the embeddings.",
    )
    argparser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Name of the dataset used for the embeddings.",
    )
    argparser.add_argument(
        "--architecture_type",
        type=str,
        required=True,
        choices=ARCHITECTURE_CHOICES,
        help="Name of the architecture used for classification.",
    )
    argparser.add_argument(
        "--fold_idx",
        type=str,
        required=False,
        default="0",
        help="Fold to be used.",
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10,
        help="Number of epochs.",
    )
    argparser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable logs on wandb."
    )
    argparser.add_argument(
        "--timestamp",
        type=str,
        required=False,
        default=datetime.now().isoformat('-', 'seconds'),
        help="Timestamp to log on wandb.",
    )
    argparser.add_argument(
        "--test",
        action="store_true",
        help="Only test the model."
    )

    args = argparser.parse_args()

    main(args)
