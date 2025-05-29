# Collectible Coins Grading

## Project Overview

This project applies deep learning techniques to the automated grading of collectible coins, using both CNNs & ViT, and CLIP-based models. The goal is to classify coins into standard grading categories based on high-resolution images of their obverse and reverse sides.

---

## Dataset

- Download from HuggingFace using `download_data.sh` (the dataset is private and requires a specialized access, ask us explicitly)
- Includes:
  - Obverse and reverse images for each coin
  - Metadata CSV: `1k-coins-dataset-no-pr.csv`
  - Grade descriptions: `Grades descriptions.csv`

Grades follow the standard MS (Mint State), AU (About Uncirculated), XF (Extremely Fine), VF (Very Fine), F (Fine), VG (Very Good), G (Good), AG (About Good), FR (Fair), PO (Poor) scale. We focuse on MS grades, see `Grades descriptions.csv` for detailed definitions.
---

## Environment Setup

1. **Clone the repository**
2. **Install dependencies**

    ```pip install -r requirements.txt```

---

## Running Experiments

### Baseline Models

- All baseline scripts are in `scripts_baseline/`.
- Experiments are defined in `scripts_baseline/experiments.yaml`.
- To run all experiments:
  ```bash
  cd scripts_baseline
  python run_experiments.py
  ```
- To run a single experiment manually:
  ```bash
  python experiment.py --model=efficientnet_b0 --feeding=channel --augmentation=baseline --learning_rate=0.0003 --batch_size=128 --run_name=example
  ```
- Checkpoints and metrics are saved in `checkpoints/` and `metrics/`.

### CLIP-based Models

- All CLIP scripts are in `scripts_clip/`.
- To fine-tune CLIP:
  ```bash
  cd scripts_clip
  python finetune_clip.py --train_tars="../1k-coins-dataset-no-pr/train-dataset-{0000..0029}.tar" --test_tars="../1k-coins-dataset-no-pr/test-dataset-{0000..0003}.tar" --meta_csv="../1k-coins-dataset-no-pr.csv" --grades_csv="../Grades descriptions.csv" --batch_size=32 --epochs=20 --lr=1e-5 --out_dir=checkpoints_clip
  ```
- You can also run separate `fusion_clip.py` (for fused image embeddings), and evaluation scripts `eval.py` & `eval_basic_clip.py`

---

## Results

Refer to our [presentation](https://docs.google.com/presentation/d/1CVD5CG28uZbv8Vkx4JBlqxHzR_xIpRIAGOdAHoBm3bo/edit?usp=sharing) for an overview of results.

## Authors
- Maxim Alymov
- Dmitry Anikin
- Roman Makarov
- Veronika Morozova
- Alexander Shmatok
