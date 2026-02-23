import argparse
import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import webdataset as wds
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import timm 
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from PIL import Image


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AdvancedAugmentation:
    """Advanced augmentation pipeline for coin defects"""
    def __init__(self, is_train=True, img_size=384):
        if is_train:
            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(45),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                # transforms.RandomAutocontrast(p=0.3),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                # transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, img):
        return self.transform(img)


class DefectEnhancement(nn.Module):
    """Enhance defect regions using learned attention"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Attention map for defect regions
        attn_map = self.attention(x)
        
        # Edge enhancement for scratch detection
        edge_map = self.edge_detector(x)
        
        # Combine and enhance
        enhanced = x * (1 + 0.5 * attn_map + 0.3 * edge_map)
        return enhanced


class SOTACoinGradeClassifier(nn.Module):
    """State-of-the-art classifier for coin grade prediction"""
    def __init__(self, model_name='convnext_xlarge', num_grades=None, pretrained=False):
        super().__init__()
        
        if num_grades is None:
            raise ValueError("num_grades must be specified")
        
        # Supported SOTA models
        self.supported_models = {
            # ConvNeXt variants (best for fine-grained visual tasks)
            'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k',
            'convnext_small': 'convnext_small.fb_in22k_ft_in1k',
            'convnext_base': 'convnext_base.fb_in22k_ft_in1k',
            'convnext_large': 'convnext_large.fb_in22k_ft_in1k',
            'convnext_xlarge': 'convnext_xlarge.fb_in22k_ft_in1k',
            
            # EfficientNet variants
            'efficientnet_b7': 'tf_efficientnet_b7.ns_jft_in1k',
            'efficientnet_b8': 'tf_efficientnet_b8.ns_jft_in1k',
            
            # ViT variants
            'vit_large': 'vit_large_patch16_384.augreg_in21k_ft_in1k',
            'vit_huge': 'vit_huge_patch14_224.in22k_ft_in22k',
            'vit_giant': 'vit_giant_patch14_224.in22k_ft_in22k',
            
            # BEiT variants
            'beit_large': 'beit_large_patch16_224.in22k_ft_in22k',
            'beitv2_large': 'beitv2_large_patch16_224.in1k',
            
            # Swin variants
            'swin_large': 'swin_large_patch4_window12_384.in22k_ft_in1k',
            'swinv2_large': 'swinv2_large_patch4_window12_192_22k',
            
            # MaxViT
            'maxvit_large': 'maxvit_large_tf_384.in1k',
            'maxvit_xlarge': 'maxvit_xlarge_tf_512.in21k_ft_in1k',
            
            # CoAtNet
            'coatnet_7': 'coatnet_7_224.nyu68c.in1k',
            
            # DINOv2 (self-supervised)
            'dinov2_giant': 'vit_giant_patch14_dinov2.lvd142m',
            
            # FocalNet
            'focalnet_large': 'focalnet_large_fl3_patch4_window7_224.ms_in1k',
            'focalnet_xlarge': 'focalnet_xlarge_fl3_patch4_window7_224.ms_in22k',
            
            # EVA
            'eva_giant': 'eva_giant_patch14_336.m30m_ft_in22k_in1k',
        }
        
        if model_name not in self.supported_models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.supported_models.keys())}")
        
        # Load backbone
        self.backbone = timm.create_model(
            self.supported_models[model_name],
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            feat_dim = self.backbone.num_features
        else:
            feat_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 2048
        
        self.feat_dim = feat_dim
        
        # Defect enhancement module
        self.defect_enhancer = DefectEnhancement(feat_dim)
        
        # Multi-scale feature fusion for obverse and reverse
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
        )
        
        # Grade-specific attention heads
        self.num_grades = num_grades
        self.grade_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim // 2, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            ) for _ in range(num_grades)
        ])

        self.grade_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim // 2, feat_dim // 2),
                nn.GELU()
            ) for _ in range(num_grades)
        ])
        
        # Main classifier with multi-head attention
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim // 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_grades)
        )
        
        # Contrastive projection head
        self.projection = nn.Sequential(
            nn.Linear(feat_dim // 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, obverse, reverse, return_features=False, return_confidence=False):
        # Extract features from both sides
        f_obv = self.backbone(obverse)
        f_rev = self.backbone(reverse)
        
        # Enhance defect features
        # if len(f_obv.shape) == 4:  # If we have spatial features
        #     f_obv = self.defect_enhancer(f_obv).mean(dim=[2, 3])
        #     f_rev = self.defect_enhancer(f_rev).mean(dim=[2, 3])
        
        # Fuse features
        fused = torch.cat([f_obv, f_rev], dim=1)
        fused = self.fusion(fused)
        
        # Grade-specific attention
        grade_attentions = []
        for i, attn_head in enumerate(self.grade_attentions):
            attn = attn_head(fused)
            grade_attentions.append(attn)
        
        grade_attention = torch.softmax(torch.cat(grade_attentions, dim=1), dim=1)
        expert_outputs = torch.stack([expert(fused) for expert in self.grade_experts], dim=1)
        
        # Weighted features for final classification
        weighted_features = (grade_attention.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        # Classification
        logits = self.classifier(weighted_features)
        
        outputs = [logits]
        
        if return_features:
            proj = self.projection(weighted_features)
            outputs.append(F.normalize(proj, dim=1))
        
        if return_confidence:
            confidence = torch.softmax(logits, dim=1).max(dim=1)[0]
            outputs.append(confidence)
        
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 2:
            return outputs[0], outputs[1]
        else:
            return outputs[0], outputs[1], outputs[2]


class HierarchicalGradeLoss(nn.Module):
    """Loss that accounts for ordinal nature of grades"""
    def __init__(self, num_grades=None, temperature=2.0):
        super().__init__()
        if num_grades is None:
            raise ValueError("num_grades must be specified")
        
        self.num_grades = num_grades
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Create distance matrix between grades
        grade_range = torch.arange(num_grades).float()
        self.distance_matrix = torch.abs(grade_range.unsqueeze(0) - grade_range.unsqueeze(1))
    
    def forward(self, logits, labels):
        # Standard CE loss with label smoothing
        ce_loss = self.ce_loss(logits, labels)
        
        # Ordinal loss - penalize more for far-off predictions
        # probs = F.softmax(logits / self.temperature, dim=1)
        
        # distance_matrix = self.distance_matrix.to(logits.device)
        # ordinal_loss = (probs.unsqueeze(2) * distance_matrix.unsqueeze(0)).sum(dim=[1, 2]).mean()
        
        return ce_loss #+ 0.5 * ordinal_loss


class ContrastiveLoss(nn.Module):
    """Enhanced contrastive loss with hard negative mining"""
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = features @ features.T / self.temperature
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_neg = torch.ne(labels, labels.T).float()
        
        # Exclude self-contrast
        mask_self = torch.eye(mask_pos.shape[0], device=mask_pos.device)
        mask_pos = mask_pos - mask_self
        
        # Compute positive and negative similarities
        pos_similarity = (similarity * mask_pos).sum(dim=1) / mask_pos.sum(dim=1).clamp(min=1)
        neg_similarity = (similarity * mask_neg).max(dim=1)[0]
        
        # Contrastive loss with margin
        loss = F.relu(neg_similarity - pos_similarity + self.margin).mean()
        
        return loss


def build_dataset(tar_pattern: str, meta_csv: Path, is_train=True, grade_to_idx=None):
    """Build dataset with advanced features"""
    id2label = {}
    id2metadata = {}
    wanted = set()
    
    # Load metadata
    df = pd.read_csv(meta_csv)
    
    for _, row in df.iterrows():
        cid, grade, split = row['id'], row['grade'], row['split']
        
        # Skip if not an MS grade
        if not grade or not isinstance(grade, str) or not grade.startswith("MS"):
            continue
        
        # Get label from mapping if provided, otherwise create on the fly
        if grade_to_idx is not None:
            if grade not in grade_to_idx:
                continue
            label = grade_to_idx[grade]
        else:
            # Convert grade to label (MS60 -> 0, MS61 -> 1, etc.)
            try:
                grade_num = int(grade[2:])
                if 60 <= grade_num <= 70:  # Allow up to MS70
                    label = grade_num - 60
                else:
                    continue
            except (ValueError, IndexError):
                continue
        
        if (is_train and split == 'train') or (not is_train and split == 'test'):
            id2label[cid] = label
            id2metadata[cid] = {
                'metal': row.get('metal', 'unknown'),
                'year': row.get('year', 0),
                'denomination': row.get('denomination', 'unknown')
            }
            wanted.add(cid)
    
    print(f"Loaded {len(wanted)} samples for {'train' if is_train else 'test'} set")
    if wanted:
        label_dist = {}
        for label in id2label.values():
            label_dist[label] = label_dist.get(label, 0) + 1
        print(f"Label distribution: {label_dist}")
    
    # Augmentation
    transform = AdvancedAugmentation(is_train=is_train)
    
    def key_filter(sample):
        return sample["__key__"] in wanted
    
    def mapper(sample):
        key = sample["__key__"]
        
        try:
            # Apply transforms
            obv = transform(sample["obverse.jpg"])
            rev = transform(sample["reverse.jpg"])
            
            # Get label and metadata
            label = id2label[key]
            metadata = id2metadata.get(key, {})
            
            return {
                'obverse': obv,
                'reverse': rev,
                'label': label,
                'metadata': metadata,
                'key': key
            }
        except Exception as e:
            print(f"Error processing sample {key}: {e}")
            return None
    
    return (
        wds.WebDataset(tar_pattern, empty_check=False)
        .decode("pil")
        .select(key_filter)
        .map(mapper)
        .select(lambda x: x is not None)
    )


def get_grade_mapping(meta_csv):
    """Get mapping from grade strings to indices (like original script)"""
    df = pd.read_csv(meta_csv)
    df_ms = df[df["grade"].str.startswith("MS")]
    unique_grades = sorted(set(df_ms["grade"].values))
    grade_to_idx = {g: i for i, g in enumerate(unique_grades)}
    idx_to_grade = {i: g for i, g in enumerate(unique_grades)}
    
    print(f"\nGrade mapping:")
    for grade, idx in grade_to_idx.items():
        print(f"  {grade} -> {idx}")
    print(f"Total grades: {len(unique_grades)}")
    
    return grade_to_idx, idx_to_grade, len(unique_grades)


def train_epoch(model, loader, optimizer, criterion_cls, criterion_cont, 
                scaler, device, epoch, alpha=0.3):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = running_cls_loss = running_cont_loss = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        obv = batch['obverse'].to(device)
        rev = batch['reverse'].to(device)
        labels = batch['label'].to(device)

        total_samples += labels.size(0)
        
        # Validate labels
        if labels.min() < 0 or labels.max() >= model.num_grades:
            print(f"\nError: Invalid labels! Min: {labels.min()}, Max: {labels.max()}")
            print(f"Labels: {labels.cpu().numpy()}")
            print(f"Expected range: [0, {model.num_grades-1}]")
            raise ValueError(f"Labels out of range")
        
        # Mixed precision training
        with autocast():
            logits, features = model(obv, rev, return_features=True)
            
            # Losses
            cls_loss = criterion_cls(logits, labels)
            cont_loss = criterion_cont(features, labels)
            loss = cls_loss + alpha * cont_loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * labels.size(0)
        running_cls_loss += cls_loss.item() * labels.size(0)
        running_cont_loss += cont_loss.item() * labels.size(0)
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if len(all_labels) > 0:
            current_acc = accuracy_score(all_labels[-100:], all_preds[-100:]) if len(all_labels) >= 10 else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.3f}'
            })
    
    # Calculate metrics
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'loss': running_loss / total_samples,
        'cls_loss': running_cls_loss / total_samples,
        'cont_loss': running_cont_loss / total_samples,
        'accuracy': train_acc,
        'f1': train_f1,
    }


def validate(model, loader, device, idx_to_grade=None):
    """Comprehensive validation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            obv = batch['obverse'].to(device)
            rev = batch['reverse'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions with confidence
            logits, confidence = model(obv, rev, return_confidence=True)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-grade metrics
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'per_class_acc': per_class_acc,
        'confidences': all_confidences,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm
    }


def train(args):
    set_seed()
    
    # Device setup
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(out_dir / 'args.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # Get grade mapping (like original script)
    grade_to_idx, idx_to_grade, num_grades = get_grade_mapping(args.meta_csv)
    
    # Build datasets with grade mapping
    print("\nBuilding datasets...")
    train_dataset = build_dataset(args.train_tars, args.meta_csv, is_train=True, grade_to_idx=grade_to_idx)
    val_dataset = build_dataset(args.test_tars, args.meta_csv, is_train=False, grade_to_idx=grade_to_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    # Initialize model with correct number of grades
    print(f"\nInitializing SOTA model: {args.model}")
    print(f"Number of grade classes: {num_grades}")
    
    model = SOTACoinGradeClassifier(
        model_name=args.model,
        num_grades=num_grades,
        pretrained=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions with correct number of grades
    criterion_cls = HierarchicalGradeLoss(num_grades=num_grades, temperature=args.temperature)
    criterion_cont = ContrastiveLoss(temperature=args.contrastive_temp, margin=args.margin)
    
    # Optimizer with layer-wise learning rate decay
    param_groups = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            # Lower learning rate for pretrained backbone
            param_groups.append({'params': param, 'lr': args.lr * 0.1, 'weight_decay': args.wd})
        else:
            # Higher learning rate for new heads
            param_groups.append({'params': param, 'lr': args.lr, 'weight_decay': args.wd})
    
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    
    # Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_acc': [], 'val_f1': [],
        'per_class_acc': [], 'lr': []
    }
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Training
        train_stats = train_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_cont,
            scaler, device, epoch, alpha=args.alpha
        )
        
        # Validation
        val_stats = validate(model, val_loader, device, idx_to_grade)
        
        # Update history
        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['accuracy'])
        history['train_f1'].append(train_stats['f1'])
        history['val_acc'].append(val_stats['accuracy'])
        history['val_f1'].append(val_stats['f1'])
        history['per_class_acc'].append(val_stats['per_class_acc'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        print(f"Train - Loss: {train_stats['loss']:.4f} "
              f"(CLS: {train_stats['cls_loss']:.4f}, "
              f"CONT: {train_stats['cont_loss']:.4f})")
        print(f"Train - Acc: {train_stats['accuracy']:.4f}, "
              f"F1: {train_stats['f1']:.4f}")
        print(f"Val   - Acc: {val_stats['accuracy']:.4f}, "
              f"F1: {val_stats['f1']:.4f}")
        
        # Per-class accuracy
        print("Per-class accuracy:")
        for i, acc in enumerate(val_stats['per_class_acc']):
            if i < len(idx_to_grade):
                grade = idx_to_grade[i]
                print(f"  {grade}: {acc:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': val_stats['accuracy'],
            'val_f1': val_stats['f1'],
            'train_stats': train_stats,
            'val_stats': val_stats,
            'history': history,
            'idx_to_grade': idx_to_grade,
            'num_grades': num_grades
        }
        
        # Save best model
        if val_stats['accuracy'] > best_val_acc:
            best_val_acc = val_stats['accuracy']
            torch.save(checkpoint, out_dir / 'best_model.pt')
            
            # Save confusion matrix for best model
            plt.figure(figsize=(12, 10))
            plt.imshow(val_stats['confusion_matrix'], interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix - Best Model')
            plt.colorbar()
            grade_labels = [idx_to_grade[i] for i in range(num_grades)]
            plt.xticks(range(num_grades), grade_labels, rotation=90)
            plt.yticks(range(num_grades), grade_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(out_dir / 'best_confusion_matrix.png', dpi=150)
            plt.close()
            
            patience_counter = 0
            print(f"✓ New best model! Val Acc: {val_stats['accuracy']:.4f}")
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if epoch % args.save_every == 0:
            torch.save(checkpoint, out_dir / f'checkpoint_epoch{epoch}.pt')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {out_dir / 'best_model.pt'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tars", type=str, 
                       default="/1k-coins-dataset-no-pr/train-dataset-{0000..0029}.tar")
    parser.add_argument("--test_tars", type=str,
                       default="/1k-coins-dataset-no-pr/test-dataset-{0000..0003}.tar")
    parser.add_argument("--meta_csv", type=Path, 
                       default="/1k-coins-dataset-no-pr.csv")
    
    # Model selection
    parser.add_argument("--model", type=str, default="convnext_xlarge")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)  # Reduced for large model
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="Weight for contrastive loss")
    parser.add_argument("--temperature", type=float, default=2.0,
                       help="Temperature for hierarchical loss")
    parser.add_argument("--contrastive_temp", type=float, default=0.07,
                       help="Temperature for contrastive loss")
    parser.add_argument("--margin", type=float, default=0.5,
                       help="Margin for contrastive loss")
    parser.add_argument("--num_workers", type=int, default=4)  # Reduced for stability
    parser.add_argument("--out_dir", type=str, default="checkpoints_sota")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda:1",
                       choices=["cuda", "cpu", "cuda:0", "cuda:1", "mps"],
                       help="Device to use: cuda, cpu, cuda:0, cuda:1, mps (for Apple Silicon)")
    
    args = parser.parse_args()
    train(args)