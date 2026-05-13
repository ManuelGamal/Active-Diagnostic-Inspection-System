#!/usr/bin/env python3
"""
Evaluate accuracy gain from RAG-enhanced inference vs full pipeline base.

Uses the actual pipeline's Stage 0 (RD++), Stage 1 (calibration), and
Stage 2 (heuristic pre-filter) for the baseline prediction.
RAG simulates learning from past ground truths.

Usage:
  cd project_root
  python -m src.rd_plus.active_diagnostic.eval_rag_accuracy
"""

import sys, os
sys.path.insert(0, '/home/manuel/Self-Supervised-Industrial-Defect-Detection-System-main/Self-Supervised-Industrial-Defect-Detection-System-main')

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from PIL import Image

DATA_DIR = Path(os.environ.get('RD_DATA_DIR', 
    '/home/manuel/Self-Supervised-Industrial-Defect-Detection-System-main/Self-Supervised-Industrial-Defect-Detection-System-main/archive (2)'))
CATEGORIES = ['bottle', 'capsule', 'carpet']

from src.rd_plus.active_diagnostic.pipeline import (
    DiagnosticState, heuristic_prefilter, _run_stage1,
    _get_all_defect_types, validate_verdict
)
from src.rd_plus.active_diagnostic.calibration import CATEGORY_STATS

# Map pipeline defect types → MVTec ground truth labels
DEFECT_LABEL_MAP = {
    'bottle': {
        'rim_crack': 'broken_large',
        'rim_chip': 'broken_small',
        'body_scratch': 'broken_large',
        'body_crack': 'broken_large',
        'contamination': 'contamination',
        'void_bubble': 'contamination',
        'label_defect': 'broken_small',
    },
    'capsule': {
        'crack': 'crack',
        'faulty_imprint': 'faulty_imprint',
        'poke': 'poke',
        'scratch': 'scratch',
        'squeeze_damage': 'crack',
        'color_variation': 'crack',
    },
    'carpet': {
        'cut': 'cut',
        'hole': 'hole',
        'color_variation': 'color',
        'thread_damage': 'cut',
        'contamination': 'hole',
    },
}

VALID_TYPES = {
    'bottle': ['broken_large', 'broken_small', 'contamination'],
    'capsule': ['crack', 'poke', 'scratch', 'faulty_imprint'],
    'carpet': ['cut', 'hole', 'color'],
}

TEST_IMAGES = [
    {'category': 'bottle', 'defect': 'broken_large', 'gt': 'broken_large'},
    {'category': 'bottle', 'defect': 'broken_small', 'gt': 'broken_small'},
    {'category': 'bottle', 'defect': 'contamination', 'gt': 'contamination'},
    {'category': 'capsule', 'defect': 'crack', 'gt': 'crack'},
    {'category': 'capsule', 'defect': 'poke', 'gt': 'poke'},
    {'category': 'capsule', 'defect': 'scratch', 'gt': 'scratch'},
    {'category': 'carpet', 'defect': 'cut', 'gt': 'cut'},
    {'category': 'carpet', 'defect': 'hole', 'gt': 'hole'},
    {'category': 'carpet', 'defect': 'color', 'gt': 'color'},
]


def load_rdpp_model(category):
    from src.rd_plus.pipeline import load_rd_model, HeatmapGenerator
    enc, proj, bn, dec = load_rd_model(category, device='cpu')
    return HeatmapGenerator(enc, proj, bn, dec, device='cpu')


def map_to_gt(pred: str, category: str) -> str:
    """Map pipeline defect type to MVTec ground truth label."""
    if pred == 'normal':
        return 'normal'
    mapping = DEFECT_LABEL_MAP.get(category, {})
    return mapping.get(pred, pred)


def run_pipeline_inference(generator, img_path, category):
    """
    Run Stages 0-2 of the pipeline.
    Returns (base_prediction, state, prefilter_result).
    """
    result = generator.forward_full(img_path)
    raw_map = result['anomaly_map']
    scale_dict = result.get('scale_maps', {})
    
    H, W = raw_map.shape
    
    # Build scale_maps as [4, H, W]
    if isinstance(scale_dict, dict):
        maps = []
        for key in ['coarse', 'medium_coarse', 'medium_fine', 'fine']:
            m = scale_dict.get(key, scale_dict.get('medium', np.zeros((H, W))))
            if m.shape != (H, W):
                from skimage.transform import resize as sk_resize
                m = sk_resize(m, (H, W), preserve_range=True)
            maps.append(m)
        scale_maps = np.stack(maps)
    else:
        scale_maps = scale_dict
    
    bbox = result.get('bbox')
    from PIL import Image
    original = np.array(Image.open(img_path).convert('RGB'))
    
    # Create DiagnosticState
    state = DiagnosticState(
        anomaly_map=raw_map,
        scale_maps=scale_maps,
        teacher_feats=[],
        category=category,
        image_path=str(img_path),
        ground_truth='',
        original_image=original,
        bbox=bbox,
    )
    
    # Stage 1: calibration
    state = _run_stage1(state)
    
    # Stage 2: heuristic pre-filter (base prediction)
    prefilter = heuristic_prefilter(state)
    
    # Base prediction = first hypothesis
    base_pred = prefilter['hypotheses'][0] if prefilter['hypotheses'] else 'normal'
    if prefilter.get('fine_z', 0) < 2.0 and prefilter.get('coarse_z', 0) < 2.0:
        base_pred = 'normal'
    
    return base_pred, state, prefilter


def rag_enhanced_pred(prefilter, state, category, rag_store):
    """
    RAG-enhanced prediction using deep embedding similarity.
    Falls back to base prediction if no good match.
    """
    hypotheses = prefilter.get('hypotheses', [])
    fine_z = prefilter.get('fine_z', 0)
    coarse_z = prefilter.get('coarse_z', 0)
    ar = prefilter.get('aspect_ratio', 1.0)
    z = state.peak_z
    
    if fine_z < 2.0 and coarse_z < 2.0 and z < 2.0:
        return 'normal', None
    
    # Query RAG for similar cases
    if category in rag_store and rag_store[category]:
        best_sim = 0
        best_gt = None
        for entry in rag_store[category]:
            sim = 1.0
            sim -= abs(entry['z'] - z) * 0.1
            sim -= abs(entry['ar'] - ar) * 0.2
            sim -= abs(entry['fine_z'] - fine_z) * 0.15
            sim -= abs(entry['coarse_z'] - coarse_z) * 0.15
            
            if sim > best_sim:
                best_sim = sim
                best_gt = entry['gt']
        
        if best_sim > 0.80:
            return best_gt, best_sim
    
    # Fall back to base
    return hypotheses[0] if hypotheses else 'unknown', None


def llm_predict(category, z, ar, fine_z, coarse_z):
    """Call Groq LLM with corrected scale data for diagnosis."""
    import json, subprocess
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None, None
    
    valid = VALID_TYPES.get(category, [])
    valid_str = ', '.join(valid)
    prompt = f"""Classify this {category} defect. Choose EXACTLY from: {valid_str}. No other words.

z={z:.1f}  AR={ar:.1f}  fine={fine_z:.1f}  coarse={coarse_z:.1f}

Rules:
- z<2.0 → output "normal"
- fine>>coarse (diff>3) → surface defect
- fine≈coarse (diff<1) → structural defect
- AR>2.5 → elongated (not <compact types>)
- AR<1.3 → compact (not <elongated types>)

JSON only: {{"defect_type": "<one of {valid_str}>", "confidence": <0.0-0.92>}}"""

    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 200,
        "response_format": {"type": "json_object"}
    })
    cmd = [
        "curl", "-s", "https://api.groq.com/openai/v1/chat/completions",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json", "-d", payload
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout:
            res = json.loads(r.stdout)
            if "error" in res:
                print(f"  [LLM] API error: {res['error']['message']}")
                return None, None
            content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                parsed = json.loads(content)
                dt = parsed.get("defect_type", "").lower().strip()
                conf = min(float(parsed.get("confidence", 0.5)), 0.92)
                if dt in [v.lower() for v in valid] or dt == 'normal':
                    return dt, conf
                print(f"  [LLM] invalid type '{dt}', valid={valid}")
                return None, None
    except Exception as e:
        print(f"  [LLM] error: {e}")
    return None, None


def evaluate():
    models = {}
    total = len(TEST_IMAGES)
    
    base_correct = 0
    rag_correct = 0
    llm_correct = 0
    llm_by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    cumulative_llm = []
    
    base_by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    rag_by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    cumulative_base = []
    cumulative_rag = []
    
    rag_store = defaultdict(list)  # category -> [{'z': ..., 'ar': ..., 'fine_z': ..., 'coarse_z': ..., 'gt': ...}]
    
    from PIL import Image
    
    print("=" * 110)
    print("RD++ Full Pipeline Evaluation: Base (Stages 0-2) vs RAG-Enhanced")
    print("=" * 110)
    print(f"\nTesting {total} images across {len(CATEGORIES)} categories\n")
    header = f"{'#':>3} {'Category':<12} {'Defect':<18} {'Ground Truth':<18} {'Base Pred':<18} {'RAG Pred':<18} {'LLM Pred':<15} {'z':>5} {'AR':>5} {'Fine':>5} {'Coarse':>5} {'B':>3} {'R':>3} {'L':>3}"
    print(header)
    print("-" * len(header))
    
    for idx, img in enumerate(TEST_IMAGES):
        cat = img['category']
        gt = img['gt']
        
        if cat not in models:
            print(f"  Loading model for {cat}...")
            models[cat] = load_rdpp_model(cat)
        
        defect_dir = img['defect']
        pngs = sorted((DATA_DIR / cat / 'test' / defect_dir).glob('*.png'))
        if not pngs:
            print(f"  WARNING: No images for {cat}/{defect_dir}")
            continue
        img_path = str(pngs[0])
        
        # Run full pipeline inference (Stages 0-2)
        base_pred, state, prefilter = run_pipeline_inference(models[cat], img_path, cat)
        
        z = state.peak_z
        fine_z = prefilter.get('fine_z', 0)
        coarse_z = prefilter.get('coarse_z', 0)
        ar = prefilter.get('aspect_ratio', 1.0)
        
        # Base accuracy
        base_pred_mapped = map_to_gt(base_pred, cat)
        base_ok = (base_pred_mapped == gt)
        if base_ok:
            base_correct += 1
        base_by_cat[cat]['correct'] += 1 if base_ok else 0
        base_by_cat[cat]['total'] += 1
        
        # RAG-enhanced prediction
        rag_pred, sim = rag_enhanced_pred(prefilter, state, cat, rag_store)
        rag_pred_mapped = map_to_gt(rag_pred, cat)
        rag_ok = (rag_pred_mapped == gt)
        if rag_ok:
            rag_correct += 1
        rag_by_cat[cat]['correct'] += 1 if rag_ok else 0
        rag_by_cat[cat]['total'] += 1
        
        # Store this case in RAG (with ground truth for future images)
        rag_store[cat].append({
            'z': z,
            'ar': ar,
            'fine_z': fine_z,
            'coarse_z': coarse_z,
            'gt': gt,
        })
        
        # LLM prediction
        llm_pred_raw, llm_conf = llm_predict(cat, z, ar, fine_z, coarse_z)
        llm_pred = llm_pred_raw if llm_pred_raw else 'N/A'
        llm_ok = (llm_pred == gt)
        if llm_pred_raw:
            llm_correct += 1 if llm_ok else 0
        llm_by_cat[cat]['total'] += 1
        llm_by_cat[cat]['correct'] += 1 if (llm_pred_raw and llm_ok) else 0
        
        cumulative_base.append(base_correct / (idx + 1) * 100)
        cumulative_rag.append(rag_correct / (idx + 1) * 100)
        cumulative_llm.append(llm_correct / (idx + 1) * 100 if llm_pred_raw else None)
        
        base_mark = "✓" if base_ok else "✗"
        rag_mark = "✓" if rag_ok else "✗"
        llm_mark = "✓" if llm_ok else ("✗" if llm_pred_raw else "-")
        sim_str = f" (sim={sim:.2f})" if sim else ""
        llm_conf_str = f"({llm_conf:.0%})" if llm_conf else ""
        print(f"{idx+1:>3} {cat:<12} {defect_dir:<18} {gt:<18} {base_pred:<10}→{base_pred_mapped:<7}{sim_str:<6} {rag_pred:<10}→{rag_pred_mapped:<7} {llm_pred:<10}{llm_conf_str:<8} {z:>5.1f} {ar:>5.1f} {fine_z:>5.1f} {coarse_z:>5.1f} {base_mark:>3} {rag_mark:>3} {llm_mark:>3}")
    
    print("-" * len(header))
    
    base_acc = base_correct / total * 100
    rag_acc = rag_correct / total * 100
    llm_n = sum(1 for v in cumulative_llm if v is not None)
    llm_acc = (llm_correct / llm_n * 100) if llm_n > 0 else 0
    gain = rag_acc - base_acc
    
    print(f"\n{'':>3} {'ACCURACY':<12} {'':<18} {'':<18} {f'{base_acc:.1f}%':<18} {f'{rag_acc:.1f}%':<18} {f'{llm_acc:.1f}%':<18} {'':>25}")
    print(f"\n  RAG Gain: {gain:+.1f}%  |  LLM Accuracy: {llm_acc:.1f}% ({llm_correct}/{llm_n})")
    
    print("\n--- By Category ---")
    for cat in CATEGORIES:
        b = base_by_cat[cat]
        r = rag_by_cat[cat]
        l = llm_by_cat[cat]
        if b['total'] > 0:
            b_acc = b['correct'] / b['total'] * 100
            r_acc = r['correct'] / r['total'] * 100
            l_acc = l['correct'] / max(l['total'], 1) * 100
            cg = r_acc - b_acc
            print(f"  {cat:<12} Base: {b_acc:5.1f}%  RAG: {r_acc:5.1f}%  LLM: {l_acc:5.1f}% ({l['correct']}/{l['total']})  Gain: {cg:+.1f}%")
    
    print("\n--- Cumulative Accuracy Curve ---")
    for i, (b, r) in enumerate(zip(cumulative_base, cumulative_rag)):
        print(f"  Image {i+1:>2}: Base={b:5.1f}%  RAG={r:5.1f}%")
    
    # Save results
    results = {
        'base_accuracy': round(base_acc, 1),
        'rag_accuracy': round(rag_acc, 1),
        'gain': round(gain, 1),
        'base_correct': base_correct,
        'rag_correct': rag_correct,
        'total': total,
        'by_category': {
            cat: {
                'base_acc': round(base_by_cat[cat]['correct'] / max(base_by_cat[cat]['total'], 1) * 100, 1),
                'rag_acc': round(rag_by_cat[cat]['correct'] / max(rag_by_cat[cat]['total'], 1) * 100, 1),
            } for cat in CATEGORIES if base_by_cat[cat]['total'] > 0
        },
        'cumulative_base': [round(v, 1) for v in cumulative_base],
        'cumulative_rag': [round(v, 1) for v in cumulative_rag],
    }
    
    out_path = Path(__file__).parent / 'eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == '__main__':
    evaluate()
