#!/usr/bin/env python3
"""
5-seed validation of active learning RAG gain.

Reproduces the +9.6% gain claim by running 5 independent trials with
different random seeds. Each trial samples images from the actual MVTec
test directories, runs base (heuristic) and RAG-enhanced prediction,
and measures accuracy gain.

Usage:
  cd <repo_root>
  python scripts/eval_seed_validation.py
"""

import sys, os, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(os.environ.get('RD_DATA_DIR',
    str(REPO_ROOT / 'archive (2)')))

from src.rd_plus.active_diagnostic.eval_rag_accuracy import (
    load_rdpp_model, run_pipeline_inference, rag_enhanced_pred,
    map_to_gt, DEFECT_LABEL_MAP, VALID_TYPES, CATEGORIES
)

SEEDS = [0, 1, 2, 3, 42]
SAMPLES_PER_DEFECT = 3  # images per defect type per trial


def build_trial_images(seed):
    """Randomly sample test images from MVTec directories using seed."""
    rng = random.Random(seed)
    trial = []
    for cat in CATEGORIES:
        valid = VALID_TYPES.get(cat, [])
        for defect_type in valid:
            img_dir = DATA_DIR / cat / 'test' / defect_type
            pngs = sorted(img_dir.glob('*.png'))
            if len(pngs) < SAMPLES_PER_DEFECT:
                print(f"  WARNING: {cat}/{defect_type} has {len(pngs)} images, need {SAMPLES_PER_DEFECT}")
                chosen = list(pngs)
            else:
                chosen = rng.sample(pngs, SAMPLES_PER_DEFECT)
            for p in chosen:
                trial.append({
                    'category': cat,
                    'defect': defect_type,
                    'gt': defect_type,
                    'path': str(p),
                })
    rng.shuffle(trial)
    return trial


def run_trial(seed, models, eval_ref_images):
    """Run a single trial with given seed. Returns trial results dict."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    trial_images = build_trial_images(seed)
    n = len(trial_images)

    base_correct = 0
    rag_correct = 0
    rag_store = defaultdict(list)

    results = []

    for idx, img in enumerate(trial_images):
        cat = img['category']
        gt = img['gt']
        img_path = img['path']

        # Ensure model loaded
        if cat not in models:
            models[cat] = load_rdpp_model(cat)

        # Run pipeline inference
        base_pred_raw, state, prefilter = run_pipeline_inference(
            models[cat], img_path, cat
        )
        base_pred = map_to_gt(base_pred_raw, cat)
        base_ok = (base_pred == gt)
        if base_ok:
            base_correct += 1

        # RAG-enhanced prediction
        rag_pred_raw, sim = rag_enhanced_pred(prefilter, state, cat, rag_store)
        rag_pred = map_to_gt(rag_pred_raw, cat)
        rag_ok = (rag_pred == gt)
        if rag_ok:
            rag_correct += 1

        # Store this image's ground truth in RAG for future images in this trial
        fine_z = prefilter.get('fine_z', 0)
        coarse_z = prefilter.get('coarse_z', 0)
        rag_store[cat].append({
            'z': state.peak_z,
            'ar': prefilter.get('aspect_ratio', 1.0),
            'fine_z': fine_z,
            'coarse_z': coarse_z,
            'gt': gt,
        })

        results.append({
            'idx': idx + 1,
            'category': cat,
            'defect': img['defect'],
            'ground_truth': gt,
            'base_pred': base_pred,
            'rag_pred': rag_pred,
            'base_ok': base_ok,
            'rag_ok': rag_ok,
        })

    base_acc = base_correct / n * 100
    rag_acc = rag_correct / n * 100
    gain = rag_acc - base_acc

    return {
        'seed': seed,
        'n_images': n,
        'base_correct': base_correct,
        'rag_correct': rag_correct,
        'base_acc': round(base_acc, 1),
        'rag_acc': round(rag_acc, 1),
        'gain': round(gain, 1),
        'positive': gain > 0,
        'results': results,
    }


def print_table(trials):
    """Print a formatted summary table."""
    sep = "-" * 100
    print(sep)
    header = f"{'Seed':>5} {'Images':>7} {'Base Corr':>9} {'RAG Corr':>9} {'Base Acc':>9} {'RAG Acc':>9} {'Gain':>7} {'Positive':>9}"
    print(header)
    print(sep)
    for t in trials:
        pos = "✓" if t['positive'] else "✗"
        print(f"{t['seed']:>5} {t['n_images']:>7} {t['base_correct']:>9} {t['rag_correct']:>9} {t['base_acc']:>8.1f}% {t['rag_acc']:>8.1f}% {t['gain']:>+6.1f}% {pos:>9}")
    print(sep)


def print_detailed(trial):
    """Print per-image detail for a trial."""
    print(f"\n  Trial seed={trial['seed']} — per-image breakdown:")
    print(f"  {'#':>3} {'Cat':<10} {'Defect':<18} {'GT':<18} {'Base':<18} {'RAG':<18} {'B':>3} {'R':>3}")
    print(f"  {'-'*90}")
    for r in trial['results']:
        bm = "✓" if r['base_ok'] else "✗"
        rm = "✓" if r['rag_ok'] else "✗"
        print(f"  {r['idx']:>3} {r['category']:<10} {r['defect']:<18} {r['ground_truth']:<18} {r['base_pred']:<18} {r['rag_pred']:<18} {bm:>3} {rm:>3}")
    print(f"\n  Accuracy: base={trial['base_acc']:.1f}%  rag={trial['rag_acc']:.1f}%  gain={trial['gain']:+.1f}%\n")


def main():
    print("=" * 80)
    print("RD++ 5-Seed RAG Gain Validation")
    print("=" * 80)
    print(f"\nSeeds: {SEEDS}")
    print(f"Categories: {CATEGORIES}")
    print(f"Samples per defect type per trial: {SAMPLES_PER_DEFECT}")
    print(f"Expected total images per trial: {len(CATEGORIES) * sum(len(VALID_TYPES[c]) for c in CATEGORIES) * SAMPLES_PER_DEFECT // len(CATEGORIES)}")
    print()

    # Pre-load models once (reuse across seeds)
    models = {}
    for cat in CATEGORIES:
        print(f"  Loading model for {cat}...")
        models[cat] = load_rdpp_model(cat)
    print()

    trials = []
    for seed in SEEDS:
        print(f"--- Trial seed={seed} ---")
        trial = run_trial(seed, models, None)
        trials.append(trial)
        print(f"  Base: {trial['base_correct']}/{trial['n_images']} = {trial['base_acc']:.1f}%")
        print(f"  RAG:  {trial['rag_correct']}/{trial['n_images']} = {trial['rag_acc']:.1f}%")
        print(f"  Gain: {trial['gain']:+.1f}%  {'✓' if trial['positive'] else '✗'}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print_table(trials)

    gains = [t['gain'] for t in trials]
    mean_gain = sum(gains) / len(gains)
    positive = sum(1 for t in trials if t['positive'])

    print(f"\n  Mean gain: {mean_gain:+.1f}%")
    print(f"  Positive trials: {positive}/{len(trials)}")
    print(f"  Per-seed gains: {[f'{g:+.1f}%' for g in gains]}")

    if 6.0 <= mean_gain <= 13.0:
        print(f"\n  ✓ Result within expected range [6%, 13%] — close to claimed +9.6%")
    else:
        print(f"\n  ⚠ Result {mean_gain:+.1f}% outside expected range [6%, 13%]")
        print(f"  Printing per-trial detail for diagnosis:")
        for t in trials:
            print_detailed(t)

    # Save results
    output = {
        'seeds': SEEDS,
        'mean_gain': round(mean_gain, 1),
        'positive_trials': positive,
        'total_trials': len(trials),
        'per_trial': [
            {
                'seed': t['seed'],
                'base_acc': t['base_acc'],
                'rag_acc': t['rag_acc'],
                'gain': t['gain'],
                'n_images': t['n_images'],
                'base_correct': t['base_correct'],
                'rag_correct': t['rag_correct'],
            }
            for t in trials
        ],
    }

    out_path = REPO_ROOT / 'scripts' / 'eval_seed_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
