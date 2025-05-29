import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_grade_prompts(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="latin1", delimiter=";")
    grades = []
    prompts = {}

    for _, row in df.iterrows():
        grade = str(row["Source"]).strip()
        if not grade.startswith("MS"):
            continue
        parts = []
        for col in df.columns:
            if col == "Source":
                continue
            val = row[col]
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    parts.append(s)
        prompt = f"{grade}: " + " ".join(parts)
        grades.append(grade)
        prompts[grade] = prompt

    grades = sorted(set(grades), key=lambda g: int(g[2:]))
    return grades, prompts


def build_dataset(tar_pattern: str, meta_csv: Path, grades: list):
    label_lookup = {g: i for i, g in enumerate(grades)}
    id2label = {}
    wanted = set()

    with open(meta_csv, encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            cols = line.split(",")
            cid, grade, split = cols[0], cols[1], cols[9]
            if not grade.startswith("MS"):
                continue
            if ("train" in tar_pattern and split != "train") or (
                "test" in tar_pattern and split != "test"
            ):
                continue
            id2label[cid] = label_lookup[grade]
            wanted.add(cid)

    norm = transforms.Normalize((0.481, 0.458, 0.408), (0.269, 0.261, 0.276))
    if "train" in tar_pattern:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                norm,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                norm,
            ]
        )

    def key_filter(sample):
        return sample["__key__"] in wanted

    def mapper(sample):
        return (
            sample["obverse.jpg"],
            sample["reverse.jpg"],
            id2label[sample["__key__"]],
        )

    return (
        wds.WebDataset(tar_pattern, empty_check=False)
        .decode("pil")
        .map_dict(**{"obverse.jpg": transform, "reverse.jpg": transform})
        .select(key_filter)
        .map(mapper)
    )


class CoinCLIP(nn.Module):
    def __init__(self, text_prompts: list, model: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)

        tokenized = self.processor(
            text=text_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        with torch.no_grad():
            te = self.clip.get_text_features(**{k: v for k, v in tokenized.items()})
            te = te / te.norm(dim=-1, keepdim=True)
        self.register_buffer("text_emb", te, persistent=False)

        dim = self.clip.config.projection_dim
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward(self, img1, img2):
        f1 = self.clip.get_image_features(img1)
        f2 = self.clip.get_image_features(img2)
        x = torch.cat([f1, f2], dim=-1)
        x = self.fusion(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x @ self.text_emb.T * self.logit_scale.exp()


def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grades, gp = load_grade_prompts(args.grades_csv)
    prompts = [gp[g] for g in grades]

    ds_tr = build_dataset(args.train_tars, args.meta_csv, grades)
    ds_te = build_dataset(args.test_tars, args.meta_csv, grades)

    train_loader = DataLoader(
        ds_tr, batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        ds_te, batch_size=args.batch_size, num_workers=8, pin_memory=True
    )

    model = CoinCLIP(prompts, args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for im1, im2, lb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            im1, im2, lb = im1.to(device), im2.to(device), lb.to(device)
            optimizer.zero_grad()
            logits = model(im1, im2)
            loss = criterion(logits, lb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * lb.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == lb).sum().item()
            total += lb.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_acc:.4f}"
        )

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for im1, im2, lb in val_loader:
                im1, im2 = im1.to(device), im2.to(device)
                logits = model(im1, im2)
                val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                val_labels.extend(lb.tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"Validation  Acc: {val_acc:.4f} " f"F1: {val_f1:.4f}")

        if args.out_dir:
            od = Path(args.out_dir)
            od.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), od / f"epoch{epoch}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_tars",
        type=str,
        default="/1k-coins-dataset-no-pr/train-dataset-{0000..0029}.tar",
    )
    p.add_argument(
        "--test_tars",
        type=str,
        default="/1k-coins-dataset-no-pr/test-dataset-{0000..0003}.tar",
    )
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--meta_csv", type=Path, default="./1k-coins-dataset-no-pr.csv")
    p.add_argument("--grades_csv", type=Path, default="./Grades descriptions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    train(p.parse_args())
