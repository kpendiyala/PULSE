# EDA_Gen (Time-Series): Finetuning MAE Backbones for Stress Prediction

This branch documents the time-series pipeline (no MFCC). It covers environment setup, data preprocessing to LOSO folds, finetuning, and multi-fold launches.

---

## 1) Environment

Conda (recommended):

```bash
conda env create -f dl.yml
conda activate dl
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

---

## 2) Repository Structure (time-series)

- `src/data/preprocess.py`: Build LOSO folds from raw CSV + WESAD pickle
- `src/data/wesad_dataset.py`: Dataset for time-series folds
- `src/model/CheapSensorMAE.py`: Time-series MAE backbone

- **Pretraining:**
  - `src/pretrain/pretrain.py`: Self-supervised pretraining for CheapSensorMAE backbones

- **Finetuning:**
  - `src/finetune/finetune.py`: Finetuning script for stress classification
  - `src/finetune/run_folds.sh`: tmux launcher for multi-fold runs

- **Knowledge Transfer (Distillation):**
  - `src/kd/train_kd.py`: Knowledge distillation (teacher-student) training script

- **Testing & Evaluation:**
  - `src/finetune/test.py`: Evaluate trained models and output metrics
  - `src/finetune/summarize_folds.py`: Aggregate and summarize results across folds

- `results/`: Outputs (models, logs) created at runtime

---

## 3) Data Preparation (LOSO time-series)

### Step 1: Download WESAD Dataset

Download the WESAD dataset:

```bash
mkdir data
cd data
wget --content-disposition "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
cd ..
```

This will download `WESAD.zip` to the `data/` directory.

### Step 2: Extract WESAD Dataset

Extract the WESAD dataset to `data/WESAD/`:

```bash
unzip data/WESAD.zip -d data/
```

This will create the directory structure: `data/WESAD/SXX/SXX.pkl` for all subjects.

### Step 3: Generate Raw CSV Files

Convert the pickle files to raw CSVs using the data wrangling script:

```bash
python src/data/data_wrangling.py --mode raw
```

This creates `data/SXX_raw_data/*.csv` directories with sensor data for each subject (ECG, BVP, EDA, temperature, accelerometer).

### Step 4: Preprocess Data into LOSO Folds

Edit `src/data/preprocess.py` to set:
- `DATA_DIR`, `WESAD_PKL_DIR`, `OUTPUT_DIR`
- `TARGET_FS`, `WINDOW_SECONDS`, `STRIDE_SECONDS`

Then run:

```bash
python -m src.data.preprocess
```

This creates `fold_*.npz` under `OUTPUT_DIR` containing train/test windows, labels and subject IDs.

---

## 4) Pretraining Backbones (Time-Series)

This branch focuses on finetuning. If you want to pretrain CheapSensorMAE backbones from scratch (self-supervised), you can:

- Train with the finetune script using `--from_scratch` and a smaller model (acts as supervised pretraining on the target task), or
- Use your own pretraining pipeline to produce a checkpoint with keys `<modality>_model_state_dict` and place it under `./results/cheap_maes/<run_name>/models/best_ckpt.pt` so finetune can load it automatically.

Example (supervised from-scratch finetune as a warm start):

```bash
python -m src.finetune.finetune \
  --run_name pretrain_like_supervised \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --from_scratch \
  --num_epochs 150 \
  --batch_size 128
```

---

## 5) Finetuning (single fold)

Minimal example (from scratch):

```bash
python -m src.finetune.finetune \
  --run_name finetune_example \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --from_scratch \
  --num_epochs 150 \
  --batch_size 128
```

Using pretrained backbones (if you have `results/cheap_maes/<run>/models/best_ckpt.pt`):

```bash
python -m src.finetune.finetune \
  --run_name finetune_pretrained \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --num_epochs 150 \
  --batch_size 128
```

Notes:
- `--modalities` accepts a comma list from `{ecg,bvp,acc,temp,eda}` or `all`.
- Use `--freeze_backbone` to train only the classifier.
- `--fuse_embeddings` and `--only_shared` control feature fusion strategy.

---

## 6) Multi-Fold Finetuning (tmux)

Edit `src/finetune/run_folds.sh`:
- `WORKDIR`, `ENV_ACTIVATE`, `RUN_NAME`, `SESSION_PREFIX`
- `FOLDS=(...)` subject IDs

Launch:

```bash
bash src/finetune/run_folds.sh
```

The script detects GPUs and assigns one per session in round-robin order. Logs go to `logs/` under `WORKDIR`.

---

## 7) Testing (evaluate runs)

Use `src/finetune/test.py` to evaluate a single run directory (containing `finetune.log` and `best_model.pt`) or a parent directory with multiple fold subdirectories. It reconstructs the model from the logged training args and checkpoint, evaluates on the test split, and writes metrics/plots.

Single run dir:

```bash
python -m src.finetune.test \
  --run_dir ./results/finetuned_models/your_run_dir \
  --device cuda:0 \
  --batch_size 256 \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --fold_number 17 \
  --modalities all
```

Parent dir (iterates over subfolders and aggregates per-fold):

```bash
python -m src.finetune.test \
  --run_dir ./results/finetuned_models/your_parent_dir \
  --device cuda:0 \
  --batch_size 256 \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --modalities all
```

Outputs per run:
- `test_metrics.json`, `test_classification_report*.txt`
- `test_confusion_matrix*.png`, `test_roc_curve.png`, `test_pr_curve.png`

Aggregates in parent dir:
- `fold_test_per_fold.csv`, `fold_test_summary.json`, `fold_test_summary.csv`

---

## 8) Tips

- Check `finetune.py --help` for all options (learning rates, scheduler restarts, embedding fusion, etc.).
- Ensure validation subject differs from the fold’s held-out test subject.
- For reproducibility, set `--seed` and keep window/stride consistent with preprocessing.

---

## 9) Knowledge Distillation (EDA teacher → CheapSensor students)

Distill an EDA MAE teacher into CheapSensorMAE students across other modalities.

```bash
python -m src.kd.train_kd \
  --run_name kd_example \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --modalities ecg,bvp,acc,temp \
  --batch_size 128 \
  --num_epochs 300 \
  --learning_rate 1e-4 \
  --mask_ratio 0.75 \
  --layers_to_match 3,5,7 \
  --kd_loss cosine \
  --lambda_hid 1.0 \
  --lambda_emb 1.0 \
  --perp_weight 0.0 \
  --recon_weight 0.0 \
  --teacher_ckpt_path ./results/eda_mae/teacher_run/models/best_ckpt.pt \
  --students_ckpt_path ./results/cheap_maes/base_run/models/best_ckpt.pt
```

Outputs:
- Full KD checkpoint: `./results/kd/<run_name>/models/best_ckpt_full_kd.pt`
- Students-only checkpoint (for finetune): `./results/kd/<run_name>/models/best_ckpt.pt`


