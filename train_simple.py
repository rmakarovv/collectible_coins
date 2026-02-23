import argparse
import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm
import webdataset as wds


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Augmentations
# ---------------------------
def build_transforms(is_train=True, img_size=384):
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])


# ---------------------------
# Dataset
# ---------------------------
def build_dataset(tar_pattern, meta_csv, is_train, grade_to_idx):
    df = pd.read_csv(meta_csv)
    wanted = {}
    split_name = "train" if is_train else "test"

    for _, row in df.iterrows():
        if not isinstance(row["grade"], str):
            continue
        if not row["grade"].startswith("MS"):
            continue
        if row["split"] != split_name:
            continue
        if row["grade"] not in grade_to_idx:
            continue
        wanted[row["id"]] = grade_to_idx[row["grade"]]

    transform = build_transforms(is_train)

    def key_filter(sample):
        return sample["__key__"] in wanted

    def mapper(sample):
        key = sample["__key__"]
        return {
            "obverse": transform(sample["obverse.jpg"]),
            "reverse": transform(sample["reverse.jpg"]),
            "label": wanted[key],
        }

    return (
        wds.WebDataset(tar_pattern, empty_check=False)
        .decode("pil")
        .select(key_filter)
        .map(mapper)
    )


# ---------------------------
# Model
# ---------------------------
class CoinGradeClassifier(nn.Module):
    def __init__(self, backbone_name, num_grades, pretrained=False):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_grades),
        )

    def forward(self, obv, rev):
        f_obv = self.backbone(obv)
        f_rev = self.backbone(rev)
        fused = torch.cat([f_obv, f_rev], dim=1)
        return self.classifier(fused)


# ---------------------------
# Training
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    running_loss = 0

    for batch in tqdm(loader, desc="Training"):
        obv = batch["obverse"].to(device)
        rev = batch["reverse"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(obv, rev)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": running_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }


# ---------------------------
# Validation with grading metrics
# ---------------------------
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            obv = batch["obverse"].to(device)
            rev = batch["reverse"].to(device)
            labels = batch["label"].to(device)

            logits = model(obv, rev)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    errors = np.abs(all_preds - all_labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "mae": errors.mean(),
        "within_1_acc": (errors <= 1).mean(),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }


# ---------------------------
# Main
# ---------------------------
def main(args):
    set_seed()
    device = torch.device(args.device)

    df = pd.read_csv(args.meta_csv)
    grades = sorted(set(g for g in df["grade"] if isinstance(g, str) and g.startswith("MS")))
    grade_to_idx = {g: i for i, g in enumerate(grades)}
    idx_to_grade = {i: g for g, i in grade_to_idx.items()}

    train_ds = build_dataset(args.train_tars, args.meta_csv, True, grade_to_idx)
    val_ds = build_dataset(args.test_tars, args.meta_csv, False, grade_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = CoinGradeClassifier(args.model, len(grades)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = validate(model, val_loader, device)

        print(f"\nEpoch {epoch}")
        print(f"Train Acc: {train_stats['accuracy']:.4f} | F1: {train_stats['f1']:.4f}")
        print(f"Val   Acc: {val_stats['accuracy']:.4f} | F1: {val_stats['f1']:.4f}")
        print(f"Val   MAE: {val_stats['mae']:.3f}")
        print(f"Val ±1 Grade Acc: {val_stats['within_1_acc']:.4f}")

        if val_stats["accuracy"] > best_acc:
            best_acc = val_stats["accuracy"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("✓ Saved best model")

    print("\nTraining complete.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tars", type=str)
    parser.add_argument("--test_tars", type=str)
    parser.add_argument("--meta_csv", type=str)
    parser.add_argument("--model", type=str, default="convnext_base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="checkpoints_baseline")

    args = parser.parse_args()
    main(args)