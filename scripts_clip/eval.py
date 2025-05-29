import argparse
import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_grade_prompts(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1", delimiter=";")
    grades, prompts = [], {}
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
        prompts[grade] = f"{grade}: " + " ".join(parts)
        grades.append(grade)
    grades = sorted(set(grades), key=lambda g: int(g[2:]))
    return grades, prompts


def build_dataset(tar_pattern: str, meta_csv: str, grades):
    label_lookup = {g: i for i, g in enumerate(grades)}
    id2label, wanted = {}, set()
    with open(meta_csv, encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            cid, grade, split = (
                line.split(",")[0],
                line.split(",")[1],
                line.split(",")[9],
            )
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
                transforms.RandomHorizontalFlip(0.5),
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

    def filt(sample):
        return sample["__key__"] in wanted

    def map_out(sample):
        return sample["obverse.jpg"], sample["reverse.jpg"], id2label[sample["__key__"]]

    return (
        wds.WebDataset(tar_pattern, empty_check=False)
        .decode("pil")
        .map_dict(**{"obverse.jpg": transform, "reverse.jpg": transform})
        .select(filt)
        .map(map_out)
    )


class CoinCLIP(nn.Module):
    def __init__(self, text_prompts):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

        proj_dim = self.clip.config.projection_dim
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoinCLIP checkpoint")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt or state_dict)",
    )
    parser.add_argument(
        "--test_tars",
        type=str,
        required=True,
        help="Pattern for test shards (e.g. '/path/to/test-*.tar')",
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        required=True,
        help="CSV with metadata for labels and splits",
    )
    parser.add_argument(
        "--grades_csv", type=str, required=True, help="CSV with grade prompts"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="metrics_fusion",
        help="Directory to save metrics",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cuda or cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.metrics_dir, exist_ok=True)

    grades, prompts = load_grade_prompts(args.grades_csv)
    prompt_list = [prompts[g] for g in grades]

    ds_test = build_dataset(args.test_tars, args.meta_csv, grades)
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, num_workers=8, pin_memory=True
    )

    model = CoinCLIP(prompt_list).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for im1, im2, lb in test_loader:
            im1, im2 = im1.to(device), im2.to(device)
            lb = lb.to(device)
            logits = model(im1, im2)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lb.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    mae = mean_absolute_error(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    proba = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    roc_auc = roc_auc_score(
        np.eye(len(grades))[all_labels], proba, multi_class="ovr", average="weighted"
    )

    with open(os.path.join(args.metrics_dir, "results.json"), "w") as f:
        json.dump(
            {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "mae": round(mae, 4),
                "Roc-Auc": round(roc_auc, 4),
            },
            f,
            indent=4,
        )
    torch.save(cm, os.path.join(args.metrics_dir, "confusion_matrix.pt"))
    np.save(os.path.join(args.metrics_dir, "confusion_matrix.npy"), cm)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    tick_marks = np.arange(len(grades))
    plt.xticks(tick_marks, grades, rotation=90)
    plt.yticks(tick_marks, grades)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(len(grades)), range(len(grades))):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(args.metrics_dir, "confusion_matrix.png"), dpi=200)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Roc-Auc: {roc_auc:.4f}")
    print(f"Saved metrics in '{args.metrics_dir}'")


if __name__ == "__main__":
    main()
