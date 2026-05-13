# Engineer 4 Acceptance Checklist

> **Scope update (Engineer 3 report):**
> Label-ratio ablation (10%/50%) was dropped. All 18 runs use **100% labels with 3-fold cross-validation**.
> Engineer 3 delivered the runs, `results.parquet`, and `04b_Ablation.ipynb`. Engineer 4 owns fold analysis,
> `best_checkpoints.json`, and `04c_FoldAnalysis.ipynb`.

---

## 1) 18-Run Execution Validation

Run naming convention (set by Engineer 3): `supervised-defect-{category}-r100-fold{N}`

```bash
python -m src.evaluation.validate_execution \
  --entity manuelaziz27-ain-shams-university \
  --project defect-detection-supervised \
  --expected-runs 18 \
  --checkpoint-manifest /path/to/kaggle_files_manifest.csv \
  --checkpoint-template "{category}_r100_f{fold}_best.ckpt"
```

**What it enforces:**

- Exactly 18 runs in scope
- All runs are `finished` (no `running` / `crashed`)
- Unique run names
- Name convention matches `supervised-defect-{category}-r100-fold{N}`
- All 6 categories × 3 folds are present (no missing combinations)
- No duplicate `(category, fold)` pairs
- Required final metrics exist: `val_auroc`, `val_loss`, `train_loss`, `epoch`
- Final epoch sanity check (accepts 0-based or 1-based final epoch logging)
- All expected checkpoints exist in the provided Kaggle manifest

---

## 2) Aggregate Output Contract

```bash
python -m src.evaluation.aggregate \
  --entity manuelaziz27-ain-shams-university \
  --project defect-detection-supervised \
  --expected-runs 18 \
  --output results/results.parquet
```

**What it enforces:**

- Produces `results/results.parquet`
- Required columns: `category`, `fold`, `run_id`, `run_name`, `val_auroc`, `val_aupr`, `val_f1`, `gpu_hours`
- `val_auroc` cannot be null (fails loudly with offending run names)
- All 6 categories × 3 folds covered
- Round-trip read check via `pandas.read_parquet()`

---

## 3) Ablation Notebook

Notebook: `notebooks/04b_Ablation.ipynb`

**What it enforces:**

- Requires `results/results.parquet` with columns: `category`, `fold`, `val_auroc`, `val_f1`, `run_id`, `run_name`
- All 6 categories present
- All 3 folds (1, 2, 3) present
- `val_auroc` has no null values
- Produces:
  - **Figure 1** — Mean Validation AUROC ± std per category (bar chart with overall mean line)
  - **Figure 2** — Per-fold AUROC line chart per category
- Exports PNGs to `results/figures/` at 200 DPI

---

## 4) Engineer 4 Deliverables (Remaining)

| Deliverable | Notebook / Script | Consumer |
|---|---|---|
| `results/fold_breakdown.csv` | `04c_FoldAnalysis.ipynb` | Eng 5 |
| `results/fold_consistency.csv` | `04c_FoldAnalysis.ipynb` | Eng 5 |
| `results/best_checkpoints.json` | `04c_FoldAnalysis.ipynb` | **Eng 6** (ONNX export blocker) |

> **Critical dependency:** Engineer 6 cannot export ONNX models until `best_checkpoints.json` is produced.
> This file maps each category to its best-fold checkpoint (`{category}_r100_f{fold}_best.ckpt`).
