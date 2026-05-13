# RD++ Active Diagnostic Defect Detection

This repository contains the **RD++ + active diagnostic** system only.
The supervised training pipeline was intentionally removed.

## Results

| Method | Score | Notes |
|---|---|---|
| RD++ Detection (AUROC) | **99.6%** | Zero defect labels, MVTec AD |
| RD++ Defect Classification (F1) | **96.9%** | Weakly supervised threshold only |
| Active learning delta | **+6.7%** | Avg across 5 seeds, 5/5 positive |

## What Makes This Different

Most industrial inspection systems are passive — one forward pass, one label, done. 
This system treats inspection as active Bayesian inference: a tool-using LLM 
issues targeted spatial queries (scale profiles, shape analysis, symmetry 
comparisons) against a cached anomaly map, eliminates hypotheses turn by turn, 
and converges on a structured verdict with root cause candidates. No equivalent 
auditable reasoning chain exists in published industrial inspection literature.

<details>
<summary>Example diagnostic chain (bottle, z=6.2σ)</summary>
Turn 1 → get_scale_profile
  fine=7.7σ >> coarse=2.2σ
  "Surface-only anomaly. Rules out structural cracks and voids."

Turn 2 → analyze_shape  
  aspect_ratio=1.9 (elongated)
  "Not circular. Rules out contamination and holes."

Verdict: body_scratch | confidence: 0.87 | severity: medium
Root cause: abrasive contact on production line
Action: inspect conveyor belt at stations 2–4
</details>

## Key Features

- RD++ anomaly map generation and evaluation
- Active diagnostic pipeline with human-in-the-loop feedback
- Streaming diagnostic UI — reasoning chain appears turn-by-turn with JET colormap heatmap overlay and active learning feedback loop
- Capable of issuing targeted spatial queries and receiving structured verdicts with reasoning chains
- RAG case store and calibration tracking
- FastAPI demo backend and HTML frontend
- Docker deployment for the RD++ demo API

## Demo

Run the server (see [Run Locally](#run-locally)) and open `http://localhost:8000/presentation.html` for the live demo with heatmap overlay and active learning UI.

## Repository Layout

- `src/rd_plus/`: RD++ core model/pipeline
- `src/rd_plus/active_diagnostic/`: API, active learning, RAG tools
- `data/`: MVTec AD dataset (download, see Setup)
- `weights/`: RD++ pretrained weights (download, see Setup)
- `utils/`, `dataset/`: RD++ runtime dependencies
- `scripts/smoke_test.py`: end-to-end API smoke test
- `run_demo.ps1`: local server launcher

## Local Setup

### 1. Download the dataset

Download MVTec AD from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place at `<repo_root>/data/`. You need at least `bottle`, `capsule`, `carpet` categories.

### 2. Download the pretrained weights

Download RD++ weights from [Google Drive](https://drive.google.com/drive/folders/1ifrkexB0N1O87CpYPS-Wg2vgAiwXFf2Z) and place in `<repo_root>/weights/`.

### 3. Install

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Run Locally

Set environment variables pointing to your dataset and weights:

```powershell
$env:RD_DATA_DIR = ".\data"
$env:RD_WEIGHTS_DIR = ".\weights"
.\run_demo.ps1
```

Server: `http://localhost:8000`

## Verify End-to-End

```powershell
.\.venv_conda\Scripts\python.exe scripts\smoke_test.py
```

Expected output:

```text
smoke_ok
```

## Docker Deployment

```powershell
cd docker
docker compose up --build
```

Then verify:

- `http://localhost:8000/health`

## Environment Variables

- `RD_DATA_DIR` — path to MVTec AD dataset (default: `<repo>/data`)
- `RD_WEIGHTS_DIR` — path to RD++ weights (default: `<repo>/weights`)
- `RD_CATEGORIES` — comma-separated categories (default: `bottle,capsule,carpet`)
- `GROQ_API_KEY` — optional, for LLM-enhanced diagnosis
- `PORT` — server port (default: `8000`)