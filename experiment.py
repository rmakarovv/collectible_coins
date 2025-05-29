import argparse
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
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
)
from pytorch_pretrained_vit import ViT
from tqdm import tqdm

import wandb

# ==== Argument Parsing ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--feeding", type=str, default="channel", choices=["channel", "width", "height", "average", "weighted"])
    parser.add_argument("--augmentation", type=str, default="baseline", choices=["baseline", "strong"])
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="experiment")
    parser.add_argument("--alpha", type=float, default=0.7, help="Alpha for weighted feeding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=128)
    return parser.parse_args()

# ==== Data Preparation ====
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
    def __init__(self, pipeline, dataset_len, transform=None, feed_strategy="channel", alpha=0.7):
        self.pipeline = pipeline
        self._len = dataset_len
        self.transform = transform or (lambda x: x)
        self.feed_strategy = feed_strategy
        self.alpha = alpha

    def __iter__(self):
        for sample in self.pipeline:
            obv = self.transform(sample["obv"])
            rev = self.transform(sample["rev"])
            if self.feed_strategy == "channel":
                img_pair = torch.cat([obv, rev], dim=0)  # (6, 224, 224)
            elif self.feed_strategy == "width":
                img_pair = torch.cat([obv, rev], dim=2)  # (3, 224, 448)
            elif self.feed_strategy == "height":
                img_pair = torch.cat([obv, rev], dim=1)  # (3, 448, 224)
            elif self.feed_strategy == "average":
                img_pair = (obv + rev) / 2
            elif self.feed_strategy == "weighted":
                img_pair = self.alpha * obv + (1 - self.alpha) * rev
            else:
                raise ValueError(f"Unknown feed strategy: {self.feed_strategy}")
            yield img_pair, sample["label"]

    def __len__(self):
        return self._len

# ==== Augmentation Configurations ====
def get_transforms(augmentation, img_size):
    if augmentation == "baseline":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform

# ==== Model Factory ====
def get_model(model_name, feed_strategy, num_classes):
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        if feed_strategy == "channel":
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
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        if feed_strategy == "channel":
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
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        if feed_strategy == "channel":
            old_conv = model.conv1
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
            model.conv1 = new_conv
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        if feed_strategy == "channel":
            old_conv = model.conv1
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
            model.conv1 = new_conv
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        # ViT expects 3-channel input, so channel concatenation is not supported here
        if feed_strategy == "channel":
            raise ValueError("ViT model does not support 6-channel input feeding strategy")
        model = ViT('B_16_imagenet1k', pretrained=True)
        # Replace classifier head
        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ==== Main Experiment Logic ====
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Initialize wandb run
    wandb.init(
        project="coin_grading_effnet",
        name=args.run_name,
        config=vars(args),
        save_code=True,
    )

    train_transform, test_transform = get_transforms(args.augmentation, args.img_size)

    train_pipeline = make_pipeline(TRAIN_SHARDS, length=train_len)
    test_pipeline = make_pipeline(TEST_SHARDS, length=test_len)

    train_dataset = CoinDataset(
        train_pipeline, dataset_len=train_len, transform=train_transform,
        feed_strategy=args.feeding, alpha=args.alpha
    )
    test_dataset = CoinDataset(
        test_pipeline, dataset_len=test_len, transform=test_transform,
        feed_strategy=args.feeding, alpha=args.alpha
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, args.feeding, len(unique_grades))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Log model gradients and topology
    wandb.watch(model, log="all", log_freq=100)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        epoch_loss = running_loss / train_len

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{args.run_name}_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": epoch_loss,
        }, ckpt_path)
        # Log checkpoint as artifact
        wandb.save(ckpt_path)

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch} [Eval]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"Epoch {epoch}: Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "test_accuracy": acc,
            "test_f1": f1,
        })

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = f"metrics/{args.run_name}_confusion_matrix_epoch_{epoch}.png"
        torch.save(cm, f"metrics/{args.run_name}_confusion_matrix_epoch_{epoch}.pt")
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest")
        plt.xticks(range(len(unique_grades)), unique_grades, rotation=90)
        plt.yticks(range(len(unique_grades)), unique_grades)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close()
        # Log confusion matrix image to wandb
        wandb.log({f"confusion_matrix_epoch_{epoch}": wandb.Image(cm_path)})

    # Log the final model checkpoint
    final_ckpt = os.path.join(ckpt_dir, f"{args.run_name}_epoch_{args.num_epochs}.pt")
    wandb.save(final_ckpt)
    # Optionally, log the model weights directly
    torch.save(model.state_dict(), f"{args.run_name}_final_model.pt")
    wandb.save(f"{args.run_name}_final_model.pt")
    wandb.finish()

if __name__ == "__main__":
    main()
