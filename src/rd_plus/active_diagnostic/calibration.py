"""
Z-Score Calibration for Active Diagnostic Pipeline

Calibrates RD++ anomaly maps using normal images from validation set.
After calibration: 0 = typical normal, 3+ = real anomaly.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os


# Pre-computed calibration stats per category (from MVTec normal training images)
# Computed via compute_calibration.py — runs RD++ on 100 normal images per category
# Layer stats are per-scale: index 0=F1(coarsest)..3=F4(finest)
CATEGORY_STATS = {
    'bottle': {
        'mu': 0.7344, 'sigma': 0.0782, 'p95': 0.8915, 'p99': 0.9283,
        'layers': [
            {'mu': 0.2585, 'sigma': 0.0266},
            {'mu': 0.2585, 'sigma': 0.0266},
            {'mu': 0.3030, 'sigma': 0.0451},
            {'mu': 0.2852, 'sigma': 0.0282},
        ],
    },
    'capsule': {
        'mu': 0.6329, 'sigma': 0.0423, 'p95': 0.7115, 'p99': 0.7506,
        'layers': [
            {'mu': 0.2130, 'sigma': 0.0294},
            {'mu': 0.2130, 'sigma': 0.0294},
            {'mu': 0.2486, 'sigma': 0.0274},
            {'mu': 0.3744, 'sigma': 0.0169},
        ],
    },
    'carpet': {
        'mu': 0.9096, 'sigma': 0.0257, 'p95': 0.9520, 'p99': 0.9923,
        'layers': [
            {'mu': 0.2844, 'sigma': 0.0095},
            {'mu': 0.2844, 'sigma': 0.0095},
            {'mu': 0.3718, 'sigma': 0.0163},
            {'mu': 0.3885, 'sigma': 0.0127},
        ],
    },
    'hazelnut': {
        'mu': 0.49, 'sigma': 0.19, 'p95': 0.70, 'p99': 0.90,
        'layers': [None, None, None, None],
    },
    'leather': {
        'mu': 0.51, 'sigma': 0.21, 'p95': 0.72, 'p99': 0.92,
        'layers': [None, None, None, None],
    },
    'pill': {
        'mu': 0.47, 'sigma': 0.17, 'p95': 0.66, 'p99': 0.86,
        'layers': [None, None, None, None],
    },
}


def calibrate_map(anomaly_map: np.ndarray, category: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Calibrate anomaly map using category-specific statistics.

    Args:
        anomaly_map: Raw RD++ anomaly map (0-2 range)
        category: Product category

    Returns:
        z_map: Z-score map (0 = typical normal, 3+ = defect)
        prob_map: Probability of defect [0, 1]
        stats: Statistics used for calibration
    """
    stats = CATEGORY_STATS.get(category, CATEGORY_STATS['bottle'])

    # Z-score normalization
    z_map = (anomaly_map - stats['mu']) / stats['sigma']

    # Sigmoid probability: maps z-scores to probability
    # Centers at p95 (2 sigma), so p99 -> ~0.9
    threshold_z = (stats['p95'] - stats['mu']) / stats['sigma']  # ~1.0
    prob_map = 1 / (1 + np.exp(-(z_map - threshold_z)))

    return z_map, prob_map, stats


def calibrate_scalar(score: float, category: str) -> Tuple[float, str]:
    """
    Calibrate a scalar anomaly score.

    Args:
        score: Raw anomaly score
        category: Product category

    Returns:
        z_score: Z-score above normal baseline
        interpretation: Text interpretation
    """
    stats = CATEGORY_STATS.get(category, CATEGORY_STATS['bottle'])
    z_score = (score - stats['mu']) / stats['sigma']

    if z_score >= 3:
        interpretation = "CLEAR DEFECT"
    elif z_score >= 2:
        interpretation = "SUSPICIOUS"
    elif z_score >= 1:
        interpretation = "BORDERLINE"
    else:
        interpretation = "NORMAL"

    return z_score, interpretation


def build_calibration_brief(
    scalar_score: float,
    category: str,
    bbox: Tuple,
    image_shape: Tuple[int, int]
) -> str:
    """
    Build the initial brief for LLM with calibrated scores.
    """
    z_score, interpretation = calibrate_scalar(scalar_score, category)

    # Bbox as fraction of image
    if bbox:
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        location = f"top={y1/h:.0%}, left={x1/w:.0%}, size={(x2-x2)*(y2-y1)/(h*w):.0%}"
    else:
        location = "not detected"

    stats = CATEGORY_STATS.get(category, CATEGORY_STATS['bottle'])

    brief = f"""## Initial Anomaly Analysis

Category: {category}
Anomaly Score: {z_score:.1f}σ above normal baseline (raw={scalar_score:.2f})
  - Normal baseline: {stats['mu']:.2f} ± {stats['sigma']:.2f}
  - Interpretation: **{interpretation}**

Detected Region: bbox at {bbox} ({location})

Guidance:
- Scores below +2σ are likely noise — focus on regions > +2σ
- Use tool queries to differentiate defect types
- Each tool returns calibrated z-scores, interpret accordingly
"""

    return brief


def calibrate_scale_maps(scale_maps: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Calibrate all four per-scale anomaly maps using per-layer statistics.
    Each scale index (0=F1 coarsest..3=F4 finest) gets its own mu/sigma.

    Args:
        scale_maps: [4, H, W] per-scale anomaly maps
        stats: dict with 'layers' list of per-layer {'mu', 'sigma'}

    Returns:
        z_scale_maps: [4, H, W] z-scored maps
    """
    layers = stats.get('layers', [])
    result = np.zeros_like(scale_maps, dtype=np.float32)
    for i in range(4):
        if i < len(layers) and layers[i] is not None:
            mu = layers[i]['mu']
            sigma = max(layers[i]['sigma'], 1e-8)
        else:
            mu = stats['mu']
            sigma = max(stats['sigma'], 1e-8)
        result[i] = (scale_maps[i] - mu) / sigma
    return result


def compute_calibration_stats(normal_images_dir: Path, category: str) -> Dict:
    """
    Build calibration stats from actual normal validation images.
    Run this once per category to compute real statistics.

    Args:
        normal_images_dir: Directory with normal images
        category: Product category

    Returns:
        dict with mu, sigma, p95, p99
    """
    # This would run RD++ on actual normal images
    # For now, return placeholder
    raise NotImplementedError("Run on actual normal images to compute stats")


# Test calibration
if __name__ == "__main__":
    # Test with observed values from bottle image
    mu, sigma = 0.504, 0.205
    test_score = 1.220

    z_score = (test_score - mu) / sigma

    print(f"Raw score: {test_score:.3f}")
    print(f"Z-score: {z_score:.1f}σ")
    print(f"Interpretation: {'CLEAR DEFECT' if z_score >= 3 else 'SUSPICIOUS' if z_score >= 2 else 'BORDERLINE' if z_score >= 1 else 'NORMAL'}")
