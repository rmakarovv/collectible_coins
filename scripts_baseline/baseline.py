import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import webdataset as wds
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

# Change the following paths to run the script
PATH_TO_CSV = "./1k-coins-dataset-no-pr.csv"
PATH_TO_DATA = "1k-coins-dataset-no-pr/"

df = pd.read_csv(PATH_TO_CSV)
df_ms = df[df["grade"].str.startswith("MS")]

df_train_ms = df_ms[df_ms["split"] == "train"]
df_test_ms = df_ms[df_ms["split"] == "test"]

id_to_grade = dict(zip(df_ms["id"], df_ms["grade"]))
unique_grades = sorted(set(id_to_grade.values()))
grade_to_idx = {g: i for i, g in enumerate(unique_grades)}
id_to_idx = {cid: grade_to_idx[grade] for cid, grade in id_to_grade.items()}

train_len = len(df_train_ms)
test_len = len(df_test_ms)

TRAIN_SHARDS = os.path.join(PATH_TO_DATA, "train-dataset-{0000..0029}.tar")
TEST_SHARDS = os.path.join(PATH_TO_DATA, "test-dataset-{0000..0003}.tar")


def make_pipeline(shard_pattern: str, length: int) -> wds.WebDataset:
    base = (
        wds.WebDataset(shard_pattern, empty_check=False)
        .with_length(length)
        .decode("pil")
    )

    def process(sample):
        cid = sample["__key__"]
        if cid not in id_to_idx:
            return None
        return {
            "obv": sample["obverse.jpg"],
            "rev": sample["reverse.jpg"],
            "label": id_to_idx[cid],
        }

    return base.map(process)


class CoinDataset(IterableDataset):
    def __init__(self, pipeline, dataset_len, transform=None):
        self.pipeline = pipeline
        self._len = dataset_len
        self.transform = transform or (lambda x: x)

    def __iter__(self):
        for sample in self.pipeline:
            obv = self.transform(sample["obv"])
            rev = self.transform(sample["rev"])
            img6 = torch.cat([obv, rev], dim=0)
            yield img6, sample["label"]

    def __len__(self):
        return self._len


def main() -> None:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_pipeline = make_pipeline(TRAIN_SHARDS, length=train_len)
    test_pipeline = make_pipeline(TEST_SHARDS, length=test_len)

    train_dataset = CoinDataset(
        train_pipeline, dataset_len=train_len, transform=train_transform
    )
    test_dataset = CoinDataset(
        test_pipeline, dataset_len=test_len, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight

    model.features[0][0] = new_conv
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(unique_grades))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
    else:
        start_epoch = 1

    num_epochs = 5
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs6, labels in tqdm(train_loader, leave=False):
            imgs6, labels = imgs6.to(device), labels.to(device)
            outputs = model(imgs6)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        epoch_loss = running_loss / train_len
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            ckpt_path,
        )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs6, labels in test_loader:
            imgs6, labels = imgs6.to(device), labels.to(device)
            outputs = model(imgs6)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("metrics", exist_ok=True)
    torch.save(cm, "metrics/confusion_matrix.pt")
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(unique_grades)), unique_grades, rotation=90)
    plt.yticks(range(len(unique_grades)), unique_grades)
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png", dpi=200)


if __name__ == "__main__":
    main()
