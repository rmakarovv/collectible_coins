import argparse
from pathlib import Path

import pandas as pd
import torch
import webdataset as wds
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

torch.backends.cudnn.benchmark = True


def load_grade_prompts(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="latin1", delimiter=";")

    grades, prompts = [], {}
    for _, row in df.iterrows():
        grade = str(row["Source"]).strip()
        if not grade.startswith("MS"):
            continue

        text_parts = []
        for col in df.columns:
            if col == "Source":
                continue
            val = row[col]
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    text_parts.append(s)
        prompt = f"{grade}: " + " ".join(text_parts)
        grades.append(grade)
        prompts[grade] = prompt

    grades = sorted(set(grades), key=lambda g: int(g[2:]))
    return grades, prompts


def build_dataset(tar_pattern: str, meta_csv: Path, grades):
    label_lookup = {g: i for i, g in enumerate(grades)}

    id2label, wanted_ids = {}, set()
    with open(meta_csv, newline="", encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            row = line.split(",")
            coin_id, grade, split = row[0], row[1], row[9]
            if not grade.startswith("MS"):
                continue
            if ("train" in tar_pattern and split != "train") or (
                "test" in tar_pattern and split != "test"
            ):
                continue
            id2label[coin_id] = label_lookup[grade]
            wanted_ids.add(coin_id)

    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    if "train" in tar_pattern:
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(20),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                # transforms.CenterCrop(224),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def id_filter(sample):
        return sample["__key__"] in wanted_ids

    def output_mapper(sample):
        return sample["obverse.jpg"], sample["reverse.jpg"], id2label[sample["__key__"]]

    return (
        wds.WebDataset(tar_pattern, empty_check=False)
        .decode("pil")
        .map_dict(**{"obverse.jpg": transform, "reverse.jpg": transform})
        .select(id_filter)
        .map(output_mapper)
    )


class CoinCLIP(nn.Module):
    def __init__(self, text_prompts):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        tokenised = self.processor(
            text=text_prompts,
            padding=True,
            truncation=True,
            max_length=77,  # https://github.com/openai/CLIP/issues/262
            return_tensors="pt",
        )
        with torch.no_grad():
            text_emb = self.clip.get_text_features(
                **{k: v for k, v in tokenised.items()}
            )
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        self.register_buffer("text_emb", text_emb, persistent=False)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward(self, img1, img2):
        feat1 = self.clip.get_image_features(img1)
        feat2 = self.clip.get_image_features(img2)
        img_feat = (feat1 + feat2) / 2
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat @ self.text_emb.T * self.logit_scale.exp()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grades, grade2prompt = load_grade_prompts(args.grades_csv)
    prompts = [grade2prompt[g] for g in grades]

    ds_train = build_dataset(args.train_tars, args.meta_csv, grades)
    ds_test = build_dataset(args.test_tars, args.meta_csv, grades)

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, num_workers=8, pin_memory=True
    )

    model = CoinCLIP(prompts).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for img1, img2, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{args.epochs} — loss: {running_loss/total:.4f} — acc: {correct/total:.4f} — lr: {current_lr:.2e}"
        )

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                preds = model(img1, img2).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            print(f"\t\t\t Val acc: {correct/total:.4f}")

        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / f"clip_coin_epoch{epoch}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune CLIP for Morgan/Peace coin MS grading"
    )
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
    p.add_argument("--meta_csv", type=Path, default="./1k-coins-dataset-no-pr.csv")
    p.add_argument("--grades_csv", type=Path, default="./Grades descriptions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    train(p.parse_args())
