#!/usr/bin/env python3
"""
Compute per-scale-layer calibration statistics from normal training images.
"""

import sys, os
BASE = '/home/manuel/Self-Supervised-Industrial-Defect-Detection-System-main/Self-Supervised-Industrial-Defect-Detection-System-main'
sys.path.insert(0, BASE)
os.chdir(BASE)

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(os.environ.get('RD_DATA_DIR', 
    '/home/manuel/Self-Supervised-Industrial-Defect-Detection-System-main/Self-Supervised-Industrial-Defect-Detection-System-main/archive (2)'))
CATEGORIES = ['bottle', 'capsule', 'carpet']


def collect_stats():
    from src.rd_plus.pipeline import load_rd_model, HeatmapGenerator
    
    results = {}
    
    for cat in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Processing {cat}...")
        print(f"{'='*60}")
        
        enc, proj, bn, dec = load_rd_model(cat, device='cpu')
        generator = HeatmapGenerator(enc, proj, bn, dec, device='cpu')
        
        img_dir = DATA_DIR / cat / 'train' / 'good'
        pngs = sorted(img_dir.glob('*.png'))
        n = min(100, len(pngs))
        print(f"  Processing {n} normal images...")
        
        fused_maxes = []
        layer_maxes = [[] for _ in range(4)]  # F1..F4
        
        for idx in range(n):
            p = pngs[idx]
            if idx % 25 == 0:
                print(f"  {idx}/{n}...")
            
            try:
                result = generator.forward_full(str(p))
                fused_maxes.append(float(result['anomaly_map'].max()))
                
                scale_dict = result.get('scale_maps', {})
                if isinstance(scale_dict, dict):
                    # forward_full maps: coarse=F2(1), medium=F3(2), fine=F4(3); no direct F1 access
                    # But we need all 4 layers. Let's map what we have:
                    # F1 (index 0) = coarsest - forward_full doesn't expose it separately
                    # We'll fall back to using the available keys and distributing
                    maps_available = {}
                    for key, val in scale_dict.items():
                        if val is not None:
                            maps_available[key] = float(np.max(val))
                    
                    if 'fine' in maps_available:
                        layer_maxes[3].append(maps_available['fine'])  # F4
                    if 'medium' in maps_available:
                        layer_maxes[2].append(maps_available['medium'])  # F3
                    if 'coarse' in maps_available:
                        layer_maxes[1].append(maps_available['coarse'])  # F2
                    # F1 not exposed - use coarse as proxy
                    if 'coarse' in maps_available:
                        layer_maxes[0].append(maps_available['coarse'])  # F1 ≈ F2
            except Exception as e:
                print(f"  Error on {p.name}: {e}")
        
        fused_arr = np.array(fused_maxes)
        results[cat] = {
            'mu': float(np.mean(fused_arr)),
            'sigma': float(np.std(fused_arr)),
            'p95': float(np.percentile(fused_arr, 95)),
            'p99': float(np.percentile(fused_arr, 99)),
            'layers': [],
            'n': len(fused_arr),
        }
        
        print(f"  Fused: mu={results[cat]['mu']:.4f}, sigma={results[cat]['sigma']:.4f}")
        
        for li in range(4):
            arr = np.array(layer_maxes[li])
            if len(arr) > 0:
                lm = float(np.mean(arr))
                ls = float(np.std(arr))
                results[cat]['layers'].append({'mu': lm, 'sigma': ls})
                print(f"  Layer F{li+1}: mu={lm:.4f}, sigma={ls:.4f}")
            else:
                results[cat]['layers'].append(None)
                print(f"  Layer F{li+1}: NO DATA")
    
    return results


def main():
    stats = collect_stats()
    
    print("\n\n" + "="*70)
    print("Copy this into calibration.py:")
    print("="*70)
    
    print(f"\nCATEGORY_STATS = {{")
    for cat, s in stats.items():
        print(f"    '{cat}': {{")
        print(f"        'mu': {s['mu']:.4f},")
        print(f"        'sigma': {s['sigma']:.4f},")
        print(f"        'p95': {s['p95']:.4f},")
        print(f"        'p99': {s['p99']:.4f},")
        print(f"        'layers': [")
        for layer in s['layers']:
            if layer:
                print(f"            {{'mu': {layer['mu']:.4f}, 'sigma': {layer['sigma']:.4f}}},")
            else:
                print(f"            None,")
        print(f"        ],")
        print(f"    }},")
    print(f"}}")
    
    out_path = Path(__file__).parent / 'calibration_data.json'
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
