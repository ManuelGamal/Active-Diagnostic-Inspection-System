"""
Active Diagnostic Pipeline - Server
FastAPI application serving RD++ analysis and active learning.
"""
from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from scipy.ndimage import label

from src.rd_plus.pipeline import HeatmapGenerator, load_rd_model
from src.rd_plus.active_diagnostic.active_learning import ActiveLearningManager, CalibrationTracker
from src.rd_plus.active_diagnostic.llm_integration import GroqDiagnosticClient
from src.rd_plus.active_diagnostic.pipeline import validate_verdict, check_confidence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv('RD_DATA_DIR', PROJECT_ROOT / 'archive (2)'))
RAG_STORE_PATH = Path(os.getenv('RD_RAG_STORE_PATH', PROJECT_ROOT / 'rag_store.json'))
CALIB_PATH = Path(os.getenv('RD_CALIB_PATH', PROJECT_ROOT / 'calibration.json'))
CATEGORIES = [c.strip() for c in os.getenv('RD_CATEGORIES', 'bottle,capsule,carpet,hazelnut,leather,pill').split(',') if c.strip()]

# Spec §Stage 3 — global confidence cap, never exceeded
MAX_CONFIDENCE = 0.92

CATEGORY_STATS = {
    'bottle':   {'mu': 0.50, 'sigma': 0.20},
    'capsule':  {'mu': 0.48, 'sigma': 0.18},
    'carpet':   {'mu': 0.52, 'sigma': 0.22},
    'hazelnut': {'mu': 0.49, 'sigma': 0.19},
    'leather':  {'mu': 0.51, 'sigma': 0.21},
    'pill':     {'mu': 0.50, 'sigma': 0.20},
}
VALID_TYPES = {
    'bottle': ['broken_large', 'contamination'],
    'capsule': ['crack', 'poke', 'scratch'],
    'carpet': ['cut', 'color', 'hole'],
    'hazelnut': ['crack', 'cut', 'hole', 'print'],
    'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
    'pill': ['color', 'crack', 'contamination', 'faulty_imprint', 'scratch']
}


class AnalyzeRequest(BaseModel):
    category: str
    image_path: str
    ground_truth: Optional[str] = None


class DiagnoseRequest(BaseModel):
    category: str
    max_z: float
    area: int
    aspect: float
    asym: float
    difficulty: str = 'high_confidence'
    image_path: Optional[str] = None
    deep_embedding: Optional[List[float]] = None


class FeedbackRequest(BaseModel):
    case_id: str
    human_answer: str
    corrected_type: Optional[str] = None


class UploadRequest(BaseModel):
    image_data: str
    category: str


# ---------------------------------------------------------------------------
# Server-side verdict helpers (mirror of pipeline.py validate_verdict)
# ---------------------------------------------------------------------------

def _heuristic_eliminated_types(category: str, aspect: float,
                               asym: float, max_z: float) -> list:
    """Populate eliminated_types in the verdict dict using shape heuristics."""
    all_types = VALID_TYPES.get(category, [])
    eliminated = []
    if aspect > 2.5:
        for t in ('contamination', 'hole', 'poke', 'void'):
            if t in all_types:
                eliminated.append(t)
    elif aspect < 1.3:
        for t in ('crack', 'scratch', 'cut', 'fold'):
            if t in all_types:
                eliminated.append(t)
    if max_z < 2.0:
        eliminated = list(all_types)   # all eliminated if below threshold
    return list(set(eliminated))


def _validate_server_verdict(verdict: dict, req) -> dict:
    """Cross-check and adjust verdict using server-side heuristics."""
    eliminated = _heuristic_eliminated_types(
        req.category, req.aspect, req.asym, req.max_z
    )
    verdict.setdefault('eliminated_types', eliminated)
    return verdict


# ---------------------------------------------------------------------------
# RD++ model loading
# ---------------------------------------------------------------------------

app = FastAPI(title='RD++ Active Diagnostic API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

models = {}
llm_client: Optional[GroqDiagnosticClient] = None
al_manager = ActiveLearningManager(
    confidence_threshold=0.75,
    rag_store_path=RAG_STORE_PATH,
    calibration_path=CALIB_PATH,
    use_llm_questions=True,
)

# Global flag to track startup
_startup_complete = False

@app.on_event('startup')
async def startup_event() -> None:
    global models, llm_client, _startup_complete
    for cat in CATEGORIES:
        models[cat] = load_rd_model(cat, device='cpu')
    try:
        llm_client = GroqDiagnosticClient()
    except Exception:
        llm_client = None
    _startup_complete = True


@app.get('/')
async def root():
    return {'status': 'RD++ Active Diagnostic API', 'docs': '/docs'}


@app.get('/presentation')
@app.get('/presentation.html')
async def presentation():
    html_path = Path(__file__).parent / 'presentation.html'
    return FileResponse(html_path, media_type='text/html') if html_path.exists() else {'status': 'not found'}


@app.get('/health')
async def health():
    return {
        'status': 'healthy' if models else 'degraded',
        'loaded_categories': sorted(list(models.keys())),
        'data_dir': str(DATA_DIR),
        'startup_complete': _startup_complete,
    }


DEMO_SAMPLES = [
    {'id': 'bottle_good', 'category': 'bottle', 'path': 'bottle/test/good/000.png', 'label': 'Bottle (Good)', 'difficulty': 'easy'},
    {'id': 'bottle_broken', 'category': 'bottle', 'path': 'bottle/test/broken_large/000.png', 'label': 'Bottle (Broken)', 'difficulty': 'human_review'},
    {'id': 'capsule_scratch', 'category': 'capsule', 'path': 'capsule/test/scratch/000.png', 'label': 'Capsule (Scratch)', 'difficulty': 'borderline'},
    {'id': 'carpet_cut', 'category': 'carpet', 'path': 'carpet/test/cut/000.png', 'label': 'Carpet (Cut)', 'difficulty': 'human_review'},
    {'id': 'hazelnut_crack', 'category': 'hazelnut', 'path': 'hazelnut/test/crack/000.png', 'label': 'Hazelnut (Crack)', 'difficulty': 'borderline'},
    {'id': 'leather_poke', 'category': 'leather', 'path': 'leather/test/poke/000.png', 'label': 'Leather (Poke)', 'difficulty': 'easy'},
    {'id': 'pill_contamination', 'category': 'pill', 'path': 'pill/test/contamination/000.png', 'label': 'Pill (Dirt)', 'difficulty': 'human_review'},
]


@app.get('/images')
async def list_images():
    images = []
    for s in DEMO_SAMPLES:
        if (DATA_DIR / s['path']).exists():
            images.append(s)
    return {'images': images}


@app.get('/presentation_sequence')
async def presentation_sequence():
    """Returns a deterministic series of 50 images for the active learning presentation."""
    import random
    random.seed(42)
    plan = [
        ("bottle",  "broken_large",  8),
        ("bottle",  "contamination", 7),
        ("capsule", "crack",         9),
        ("capsule", "poke",          8),
        ("capsule", "scratch",       8),
        ("carpet",  "cut",           5),
        ("carpet",  "hole",          5),
    ]
    sequence = []
    for category, defect_dir, count in plan:
        available = sorted((DATA_DIR / category / "test" / defect_dir).glob("*.png"))
        for p in available[:count]:
            sequence.append({
                'id': f"{category}_{defect_dir}_{p.stem}",
                'category': category,
                'path': str(p.relative_to(DATA_DIR)).as_posix(),
                'label': f"{category} ({defect_dir})",
                'difficulty': 'borderline' if defect_dir in ('scratch', 'hole') else 'human_review'
            })
    random.shuffle(sequence)
    return {'sequence': sequence[:50], 'total': len(sequence)}


@app.get('/image/{img_id}')
async def get_image_by_id(img_id: str):
    mapping = {s['id']: s['path'] for s in DEMO_SAMPLES}
    if img_id not in mapping:
        raise HTTPException(status_code=404, detail="Image ID not found")
    img_path = DATA_DIR / mapping[img_id]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(img_path, media_type='image/png')


@app.get('/raw_image/{path:path}')
async def get_raw_image(path: str):
    img_path = DATA_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    with open(img_path, 'rb') as f:
        img_data = f.read()
    return {'image': "data:image/png;base64," + base64.b64encode(img_data).decode('utf-8')}


@app.get('/get_image')
async def get_image(path: str):
    """Serve image by path - used by presentation HTML."""
    img_path = Path(path)
    if not img_path.is_absolute():
        img_path = DATA_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")
    with open(img_path, 'rb') as f:
        img_data = f.read()
    return {"image": "data:image/png;base64," + base64.b64encode(img_data).decode('utf-8')}


@app.get('/presentation_image')
async def presentation_image(path: str):
    """Serve image directly for presentation HTML."""
    img_path = Path(path)
    if not img_path.is_absolute():
        img_path = DATA_DIR / path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")
    return FileResponse(img_path, media_type='image/png')


@app.post('/analyze')
async def analyze(req: AnalyzeRequest):
    if req.category not in models:
        raise HTTPException(status_code=400, detail=f'Unknown category: {req.category}')

    candidate_path = Path(req.image_path)
    img_path = candidate_path if candidate_path.is_absolute() else (DATA_DIR / req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f'Image not found: {img_path}')

    enc, proj, bn, dec = models[req.category]
    generator = HeatmapGenerator(enc, proj, bn, dec, device='cpu')
    result = generator.forward_full(img_path)

    raw_map = result['anomaly_map']
    stats = CATEGORY_STATS.get(req.category, {'mu': 0.5, 'sigma': 0.2})
    z_map = (raw_map - stats['mu']) / max(stats['sigma'], 1e-8)

    max_z = float(z_map.max())
    # SPEC: threshold must be in z-score space at 2.0σ, never raw score space.
    # Using raw_map > 0.3 is category-dependent and produces meaningless regions.
    binary = z_map > 2.0
    area = int(binary.sum())

    aspect = 1.0
    labeled, num = label(binary)
    if num > 0 and area > 10:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        comp = labeled == (int(np.argmax(sizes)) + 1)
        # Compute aspect ratio via PCA on pixel coordinates (robust vs bbox heuristic)
        coords = np.array(np.where(comp)).T  # (N, 2)
        if len(coords) > 5:
            centered    = coords - coords.mean(axis=0)
            cov         = np.cov(centered.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
            aspect = float(np.sqrt(eigenvalues[0] / (eigenvalues[1] + 1e-8)))
        else:
            rows = np.any(comp, axis=1)
            cols = np.any(comp, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                h_box = rmax - rmin + 1
                w_box = cmax - cmin + 1
                aspect = max(h_box, w_box) / max(1, min(h_box, w_box))

    import matplotlib.pyplot as plt

    H, W = z_map.shape
    left = float(z_map[:, :W // 2].mean())
    right = float(z_map[:, W // 2:].mean())
    asym = (max(left, right) + 1e-8) / (min(left, right) + 1e-8)

    # Generate heatmap overlay with matplotlib
    img_raw = Image.open(img_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_raw)
    ax.imshow(z_map, cmap='jet', alpha=0.5, vmin=0, vmax=max(3.0, max_z))
    ax.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    interpretation = "Defect detected" if max_z > 3.0 else "Normal"

    return {
        'category': req.category,
        'image_path': req.image_path,
        'max_z': max_z,
        'area': area,
        'aspect': aspect,
        'aspect_ratio': aspect,
        'asym': asym,
        'asymmetry': asym,
        'interpretation': interpretation,
        'heatmap_image': heatmap_b64,
    }


@app.post('/diagnose_simple')
async def diagnose_simple(req: DiagnoseRequest):
    valid = VALID_TYPES.get(req.category, [])
    
    SYSTEM = """
You are an industrial defect classifier. You MUST follow this exact process:

Step 1 — Call get_scale_profile first. Always.
  fine >> coarse (ratio > 3): ELIMINATE crack, void, structural defects
  fine ≈ coarse (ratio < 2):  ELIMINATE contamination, stain, color

Step 2 — Call analyze_shape second. Always.
  aspect_ratio > 2.0: ELIMINATE contamination, hole, poke, color_stain
  aspect_ratio < 1.5: ELIMINATE crack, scratch, cut

Step 3 — State what types REMAIN after elimination.
  Output verdict from remaining types only.
  If only one type remains, confidence = 0.85.
  If two types remain, confidence = 0.65 → triggers human review.
  Never output contamination if aspect_ratio > 2.0.
  Never output color if fine/coarse ratio > 3.0.

Max confidence: 0.88. Never output 1.0.

Respond in exactly this JSON format:
{
  "reasoning_chain": [
    "TOOL_CALL: get_scale_profile() -> ratio=...",
    "ELIMINATED: ...",
    "TOOL_CALL: analyze_shape() -> aspect_ratio=...",
    "ELIMINATED: ...",
    "REMAINING: ..."
  ],
  "defect_type": "...",
  "confidence": 0.65
}
"""

    predicted = 'unknown'
    confidence = 0.60
    reasoning_chain = []
    
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key and req.max_z >= 1.5:
        import json
        import subprocess
        
        user_msg = (
            f"Category: {req.category}. Valid defect types: {', '.join(valid)}. "
            f"Analysis: z-score={req.max_z:.1f}sigma, area={req.area}px, "
            f"aspect_ratio={req.aspect:.1f}, asymmetry={req.asym:.1f}."
        )
        
        payload = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        })
        
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            cmd = [
                "curl", "-s", "https://api.groq.com/openai/v1/chat/completions",
                "-H", f"Authorization: Bearer {api_key}",
                "-H", "Content-Type: application/json",
                "-d", payload
            ]
            
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if r.returncode == 0 and r.stdout:
                    res = json.loads(r.stdout)
                    content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                    parsed = json.loads(content)
                    if "defect_type" in parsed:
                        predicted = parsed["defect_type"]
                        confidence = parsed.get("confidence", 0.65)
                        reasoning_chain = parsed.get("reasoning_chain", [])
            except Exception:
                import traceback
                
    # Fallback: heuristic prediction when LLM fails or is unavailable
    if predicted == 'unknown' and req.max_z >= 2.0:
        if req.aspect > 2.5:
            predicted = 'crack' if 'crack' in valid else (valid[0] if valid else 'unknown')
        elif req.aspect < 1.3:
            predicted = 'contamination' if 'contamination' in valid else (valid[0] if valid else 'unknown')
        else:
            predicted = valid[0] if valid else 'unknown'
        confidence = min(0.75, max(0.55, req.max_z / 5.0))
        reasoning_chain = [
            f"Heuristic: z={req.max_z:.1f}σ, AR={req.aspect:.1f}, asym={req.asym:.1f}",
            f"Predicted: {predicted} (confidence {confidence:.0%})"
        ]

    return {
        'status': 'complete',
        'case_id': 'mock_case_' + req.category,
        'verdict': {
            'defect_type': predicted,
            'confidence': round(confidence, 2),
            'eliminated_types': valid,
            'action': 'flag_human' if confidence < 0.85 else 'auto_accept',
            'question': 'Is this a ' + predicted + '?' if confidence < 0.85 else None,
        },
    }


def _run_llm_diagnosis(category: str, max_z: float, area: int, aspect: float, asym: float) -> Tuple[str, float, List[str]]:
    """Run LLM diagnosis via Groq API. Returns (predicted_type, confidence, reasoning_chain).
    Falls back to heuristic when LLM unavailable or fails."""
    valid = VALID_TYPES.get(category, [])
    predicted = 'unknown'
    confidence = 0.60
    reasoning_chain = []

    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key and max_z >= 1.5:
        print(f"[Dx] Calling Groq LLM for {category} (z={max_z:.1f})...")
        SYSTEM = """
You are an industrial defect classifier. You MUST follow this exact process:
Step 1 — Call get_scale_profile first. Always.
Step 2 — Call analyze_shape second. Always.
Step 3 — Eliminate IMPOSSIBLE types based on tool evidence.
Step 4 — Output verdict from remaining types only.
Max confidence: 0.88. Never output 1.0.
Respond in JSON: {"defect_type": "...", "confidence": 0.65, "reasoning_chain": [...]}
"""
        import json, subprocess
        user_msg = (
            f"Category: {category}. Valid types: {', '.join(valid)}. "
            f"z={max_z:.1f}σ, area={area}px, AR={aspect:.1f}, asym={asym:.1f}."
        )
        payload = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}],
            "temperature": 0.1, "max_tokens": 500, "response_format": {"type": "json_object"}
        })
        cmd = [
            "curl", "-s", "https://api.groq.com/openai/v1/chat/completions",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json", "-d", payload
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout:
                res = json.loads(r.stdout)
                if "error" in res:
                    print(f"[Dx] Groq API error: {res['error']}")
                else:
                    content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        parsed = json.loads(content)
                        if "defect_type" in parsed:
                            predicted = parsed["defect_type"]
                            confidence = min(parsed.get("confidence", 0.65), MAX_CONFIDENCE)
                            reasoning_chain = parsed.get("reasoning_chain", [])
                            print(f"[Dx] LLM OK: {predicted} (confidence={confidence:.0%})")
        except Exception as e:
            print(f"[Dx] LLM failed: {e}")
    else:
        print(f"[Dx] No LLM (api_key={'set' if api_key else 'missing'}, z={max_z:.1f})")

    # Fallback heuristic if LLM fails
    if predicted == 'unknown' and max_z >= 2.0:
        if aspect > 2.5:
            predicted = valid[0] if valid else 'unknown'
        elif aspect < 1.3:
            predicted = valid[1] if len(valid) > 1 else (valid[0] if valid else 'unknown')
        else:
            predicted = valid[0] if valid else 'unknown'
        confidence = min(0.75, max(0.55, max_z / 5.0))
        reasoning_chain = [f"Heuristic: z={max_z:.1f}, AR={aspect:.1f} -> {predicted}"]

    return predicted, confidence, reasoning_chain


def _generate_heatmap(img_path, z_map, max_z):
    import matplotlib.pyplot as plt
    from PIL import Image
    img_raw = Image.open(img_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_raw)
    ax.imshow(z_map, cmap='jet', alpha=0.5, vmin=0, vmax=max(3.0, max_z))
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')


def _build_response(al_result, max_z, heatmap_b64, verdict, action_label, question, rag_matched, ground_truth):
    correct = None
    if ground_truth:
        correct = verdict['defect_type'].lower() == ground_truth.lower()
        verb = "✓" if correct else "✗"
        print(f"[GT] {ground_truth} vs predicted={verdict['defect_type']} → {verb}")
    return {
        'status': 'complete',
        'case_id': al_result.get('case_id', f'case_{max_z:.1f}'),
        'max_z': max_z,
        'heatmap_image': heatmap_b64,
        'verdict': {
            'defect_type': verdict['defect_type'],
            'confidence': round(verdict['confidence'], 2),
            'severity': verdict.get('severity', 'medium'),
            'eliminated_types': verdict.get('eliminated_types', []),
            'action': action_label,
            'question': question,
        },
        'rag_matched': rag_matched,
        'ground_truth': ground_truth,
        'correct': correct,
    }


@app.post('/analyze_simple')
async def analyze_simple(req: AnalyzeRequest):
    """Ask human about every image. No auto-accept, no LLM."""
    if req.category not in models:
        raise HTTPException(status_code=400, detail=f'Unknown category: {req.category}')

    candidate_path = Path(req.image_path)
    img_path = candidate_path if candidate_path.is_absolute() else (DATA_DIR / req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f'Image not found: {img_path}')

    # --- RD++ analysis ---
    enc, proj, bn, dec = models[req.category]
    generator = HeatmapGenerator(enc, proj, bn, dec, device='cpu')
    result = generator.forward_full(img_path)

    raw_map = result['anomaly_map']
    stats = CATEGORY_STATS.get(req.category, {'mu': 0.5, 'sigma': 0.2})
    z_map = (raw_map - stats['mu']) / max(stats['sigma'], 1e-8)
    max_z = float(z_map.max())
    binary = z_map > 2.0
    area = int(binary.sum())

    aspect = 1.0
    labeled, num = label(binary)
    if num > 0 and area > 10:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        comp = labeled == (int(np.argmax(sizes)) + 1)
        coords = np.array(np.where(comp)).T
        if len(coords) > 5:
            centered = coords - coords.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
            aspect = float(np.sqrt(eigenvalues[0] / (eigenvalues[1] + 1e-8)))

    H, W = z_map.shape
    left = float(z_map[:, :W // 2].mean())
    right = float(z_map[:, W // 2:].mean())
    asym = (max(left, right) + 1e-8) / (min(left, right) + 1e-8)

    # --- Build simple verdict based on z-score only ---
    if max_z >= 3.0:
        predicted = 'defect'
    elif max_z >= 2.0:
        predicted = 'suspicious'
    else:
        predicted = 'normal'

    verdict = {
        'defect_type': predicted,
        'confidence': round(min(0.75, max_z / 5.0), 2),
        'severity': 'high' if max_z >= 4.0 else ('medium' if max_z >= 2.5 else 'low'),
        'location': 'detected region',
        'eliminated_types': _heuristic_eliminated_types(req.category, aspect, asym, max_z),
        'root_cause_candidates': [],
        'recommended_action': 'Human review required',
        'reasoning_summary': f'z={max_z:.1f}σ, AR={aspect:.1f}, area={area}px',
        'unresolved_uncertainty': 'Human review needed',
    }

    # --- Generate question ---
    if max_z >= 3.0:
        question = f"z={max_z:.1f}σ — strong anomaly in this {req.category}. Is this a defect?"
    elif max_z >= 2.0:
        question = f"z={max_z:.1f}σ — suspicious region in this {req.category}. Is this a defect?"
    else:
        question = f"z={max_z:.1f}σ — no strong signal. Can you see any defect in this {req.category}?"

    # --- Register in active learning (no auto-store, human decides) ---
    state = {
        'category': req.category, 'max_z': max_z, 'area': area,
        'aspect_ratio': aspect, 'asymmetry': asym, 'image_path': req.image_path,
    }
    al_result = al_manager.run(verdict, state)

    heatmap_b64 = _generate_heatmap(img_path, z_map, max_z)
    return _build_response(
        al_result, max_z, heatmap_b64, verdict, 'flag_human',
        question, False, req.ground_truth
    )


@app.post('/feedback')
async def feedback(req: FeedbackRequest):
    fb = al_manager.incorporate_feedback(
        case_id=req.case_id,
        human_answer=req.human_answer,
        corrected_type=req.corrected_type,
    )
    return {
        'status': 'recorded',
        'case_id': req.case_id,
        'human_answer': req.human_answer,
        'corrected_type': fb.corrected_type,
        'rag_total_cases': len(al_manager.rag),
    }


@app.post('/upload_analyze')
async def upload_analyze(req: UploadRequest):
    if req.category not in models:
        raise HTTPException(status_code=400, detail=f'Unknown category: {req.category}')

    img_data = base64.b64decode(req.image_data.split(',')[1])
    img = Image.open(BytesIO(img_data)).convert('RGB')

    with NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = Path(tmp.name)

    enc, proj, bn, dec = models[req.category]
    generator = HeatmapGenerator(enc, proj, bn, dec, device='cpu')
    result = generator.forward_full(tmp_path)

    raw_map = result['anomaly_map']
    stats = CATEGORY_STATS.get(req.category, {'mu': 0.5, 'sigma': 0.2})
    z_map = (raw_map - stats['mu']) / max(stats['sigma'], 1e-8)

    max_z = float(z_map.max())
    binary = z_map > 2.0
    area = int(binary.sum())

    aspect = 1.0
    labeled, num = label(binary)
    if num > 0 and area > 10:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        comp = labeled == (int(np.argmax(sizes)) + 1)
        coords = np.array(np.where(comp)).T
        if len(coords) > 5:
            centered = coords - coords.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
            aspect = float(np.sqrt(eigenvalues[0] / (eigenvalues[1] + 1e-8)))
        else:
            rows = np.any(comp, axis=1)
            cols = np.any(comp, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                h_box = rmax - rmin + 1
                w_box = cmax - cmin + 1
                aspect = max(h_box, w_box) / max(1, min(h_box, w_box))

    H, W = z_map.shape
    left = float(z_map[:, :W // 2].mean())
    right = float(z_map[:, W // 2:].mean())
    asym = (max(left, right) + 1e-8) / (min(left, right) + 1e-8)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(z_map, cmap='jet', alpha=0.5, vmin=0, vmax=max(3.0, max_z))
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    interpretation = "Defect detected" if max_z > 3.0 else "Normal"

    os.unlink(tmp_path)

    return {
        'category': req.category,
        'max_z': max_z,
        'area': area,
        'aspect': aspect,
        'aspect_ratio': aspect,
        'asym': asym,
        'asymmetry': asym,
        'interpretation': interpretation,
        'heatmap_image': heatmap_b64,
    }


@app.get('/rag/summary')
async def rag_summary():
    recal = {}
    for cat in CATEGORY_STATS:
        stats = al_manager.get_recalibrated_stats(cat)
        if stats:
            recal[cat] = stats
    return {
        'rag': al_manager.rag_summary(),
        'recalibrated_categories': recal,
        'pending_reviews': len(al_manager.get_pending()),
    }


@app.get('/rag/recalibration')
async def rag_recalibration():
    updates = {}
    for cat in CATEGORY_STATS:
        stats = al_manager.get_recalibrated_stats(cat)
        if stats:
            updates[cat] = {
                'current_mu': CATEGORY_STATS[cat]['mu'],
                'current_sigma': CATEGORY_STATS[cat]['sigma'],
                'learned_mu': stats['mu'],
                'learned_sigma': stats['sigma'],
                'n_samples': stats['n'],
                'source': stats['source'],
            }
    return {
        'recalibrated': updates,
        'note': f"{CalibrationTracker.RETRAIN_THRESHOLD} human samples per category required.",
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))