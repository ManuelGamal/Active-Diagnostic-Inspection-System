"""
Tool Execution Layer for Active Diagnostic Pipeline
All tools operate on cached anomaly maps - no re-inference required.

Key invariant: every tool operates in z-score space wherever a threshold is
applied.  Raw anomaly_map scores are category-dependent and must never be
threshold-compared directly.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


def interpret_z(z: float) -> str:
    """Interpret z-score for LLM."""
    if z < 1.5:
        return "normal (noise)"
    elif z < 2.5:
        return "suspicious"
    elif z < 3.5:
        return "likely defect"
    else:
        return "clear defect"


# Scale-map index constants (per spec)
_SCALE_IDX = {
    "coarse":       0,  # F1 — structural (coarsest)
    "medium_coarse": 1, # F2
    "medium_fine":  2,  # F3
    "fine":         3,  # F4 — surface texture (finest)
    "medium":       2,  # alias -> F3
}


class ToolExecutor:
    """
    Executes spatial queries on cached, calibrated anomaly maps.

    Requires state to contain z_map and z_scale_maps (populated by Stage 1).
    Tools return explicit errors rather than silently falling back to raw scores.
    """

    def __init__(self, state: Dict):
        self.state        = state
        self.anomaly_map  = state['anomaly_map']
        self.z_map        = state.get('z_map')           # [H, W] calibrated z-scores
        self.z_scale_maps = state.get('z_scale_maps')    # [4, H, W] per-scale z-scores
        self.prob_map     = state.get('prob_map')
        self.H, self.W    = self.anomaly_map.shape
        self._region_masks = self._build_region_masks()

    # ---------------------------------------------------------------- masks

    def _build_region_masks(self) -> Dict[str, np.ndarray]:
        H, W = self.H, self.W
        return {
            'top_half':    self._rect(0,     0,     H // 2, W),
            'bottom_half': self._rect(H // 2, 0,    H,      W),
            'left_half':   self._rect(0,     0,     H,      W // 2),
            'right_half':  self._rect(0,     W // 2, H,     W),
            'center':      self._rect(H // 4, W // 4, 3*H//4, 3*W//4),
            'full':        np.ones((H, W), dtype=bool),
        }

    def _rect(self, y1, x1, y2, x2) -> np.ndarray:
        m = np.zeros((self.H, self.W), dtype=bool)
        m[y1:y2, x1:x2] = True
        return m

    def _get_region_mask(self, region: str, bbox=None) -> np.ndarray:
        if region in self._region_masks:
            return self._region_masks[region]
        bbox_val = bbox or self.state.get('bbox')
        if bbox_val is not None:
            x1, y1, x2, y2 = bbox_val
            margin = 10
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
            x2 = min(self.W, x2 + margin); y2 = min(self.H, y2 + margin)
            if region == 'bbox_interior':
                return self._rect(y1, x1, y2, x2)
            if region == 'bbox_boundary':
                outer = self._rect(y1, x1, y2, x2)
                inner = self._rect(y1+5, x1+5, y2-5, x2-5)
                return outer & ~inner
        return self._region_masks['full']

    # ---------------------------------------------------------------- Tool 1

    def query_region(
        self,
        region: str = "full",
        scale:  str = "all",
        aggregate: str = "max",
        bbox=None,
    ) -> ToolResult:
        """
        Tool 1: WHERE is the defect and HOW BAD in a specific region?

        scale='fine'   reads z_scale_maps[3] (F4, surface texture)
        scale='coarse' reads z_scale_maps[0] (F1, structural)
        scale='all'    reads z_map (fused)
        """
        try:
            mask = self._get_region_mask(region, bbox)
            if mask.sum() == 0:
                return ToolResult("query_region", False, {}, "Empty region")

            # Route to correct source map
            if scale == "all" or scale not in _SCALE_IDX:
                source = self.z_map if self.z_map is not None else self.anomaly_map
            elif self.z_scale_maps is not None:
                source = self.z_scale_maps[_SCALE_IDX[scale]]
            else:
                source = self.z_map if self.z_map is not None else self.anomaly_map

            vals = source[mask]
            agg_fns = {
                "max":           lambda v: float(v.max()),
                "mean":          lambda v: float(v.mean()),
                "p95":           lambda v: float(np.percentile(v, 95)),
                "area_fraction": lambda v: float((v > 2.0).mean()),
            }
            value = agg_fns.get(aggregate, agg_fns["max"])(vals)

            return ToolResult("query_region", True, {
                "z_score":        round(value, 2),
                "interpretation": interpret_z(value),
                "region":         region,
                "scale":          scale,
                "aggregate":      aggregate,
                "pixel_count":    int(mask.sum()),
            })
        except Exception as e:
            return ToolResult("query_region", False, {}, str(e))

    # ---------------------------------------------------------------- Tool 2

    def analyze_shape(self, z_threshold: float = 2.0) -> ToolResult:
        """
        Tool 2: WHAT SHAPE is the defect?

        CRITICAL: thresholds in z-score space (z_map > z_threshold=2.0).
        Never threshold on raw anomaly_map — scores are category-dependent
        and produce meaningless regions.

        Returns:
          aspect_ratio  >3.0 elongated (crack/scratch), ~1.0 compact (hole/contamination)
          n_components  number of disconnected regions above threshold
          area_pixels   pixels above threshold in largest component
          touches_edge  whether largest component touches image boundary
          interpretation hard-elimination guidance string
        """
        try:
            if self.z_map is None:
                return ToolResult("analyze_shape", False, {},
                    "z_map not available — Stage 1 calibration must run before tools")

            binary  = self.z_map > z_threshold
            labeled, num = label(binary)

            if num == 0:
                return ToolResult("analyze_shape", True, {
                    "aspect_ratio":   1.0,
                    "n_components":   0,
                    "area_pixels":    0,
                    "touches_edge":   False,
                    "interpretation": "no region above threshold — weak or absent defect",
                    "z_threshold":    z_threshold,
                })

            sizes = [int((labeled == i).sum()) for i in range(1, num + 1)]
            largest_label = int(np.argmax(sizes)) + 1
            component = (labeled == largest_label)

            # Aspect ratio via PCA on pixel coordinates
            coords = np.array(np.where(component)).T
            if len(coords) > 5:
                centered    = coords - coords.mean(axis=0)
                cov         = np.cov(centered.T)
                eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
                aspect_ratio = float(np.sqrt(eigenvalues[0] / (eigenvalues[1] + 1e-8)))
            else:
                aspect_ratio = 1.0

            touches_edge = bool(
                component[0, :].any() or component[-1, :].any() or
                component[:, 0].any() or component[:, -1].any()
            )

            if aspect_ratio > 3.0:
                interp = "elongated — RULES OUT contamination, hole, poke, void"
            elif aspect_ratio > 1.8:
                interp = "moderately elongated — ambiguous, check scale profile"
            else:
                interp = "compact — RULES OUT crack, scratch, cut, fold"
            if num > 3:
                interp += "; multiple components — RULES OUT localized defects"

            return ToolResult("analyze_shape", True, {
                "aspect_ratio":   round(aspect_ratio, 2),
                "n_components":   num,
                "area_pixels":    sizes[largest_label - 1],
                "touches_edge":   touches_edge,
                "interpretation": interp,
                "z_threshold":    z_threshold,
            })
        except Exception as e:
            return ToolResult("analyze_shape", False, {}, str(e))

    # ---------------------------------------------------------------- Tool 3

    def compare_symmetric(self, axis: str = "vertical") -> ToolResult:
        """
        Tool 3: IS the defect one-sided or symmetric?

        Uses z_map means (not max) to avoid single-pixel noise dominating.
        ratio > 3.0 = strongly asymmetric (localized)
        ratio 1.5-3.0 = moderate asymmetry
        ratio < 1.5 = symmetric / centered
        """
        try:
            if self.z_map is None:
                return ToolResult("compare_symmetric", False, {},
                    "z_map not available — Stage 1 calibration must run first")

            if axis == "vertical":
                a_map, b_map = self.z_map[:, :self.W//2], self.z_map[:, self.W//2:]
                la, lb = "left", "right"
            else:
                a_map, b_map = self.z_map[:self.H//2, :], self.z_map[self.H//2:, :]
                la, lb = "top", "bottom"

            za = float(a_map.mean())
            zb = float(b_map.mean())
            dominant = la if za > zb else lb
            ratio    = max(za, zb) / (min(za, zb) + 1e-8)

            if ratio > 3.0:
                interp = f"strongly asymmetric — defect on {dominant} side, rules out centered defects"
            elif ratio > 1.5:
                interp = f"moderately asymmetric — slight {dominant} concentration"
            else:
                interp = "symmetric — centered or diffuse defect, rules out edge/rim damage"

            return ToolResult("compare_symmetric", True, {
                f"z_{la}":        round(za, 2),
                f"z_{lb}":        round(zb, 2),
                "asymmetry_ratio": round(ratio, 2),
                "dominant_side":  dominant,
                "interpretation": interp,
            })
        except Exception as e:
            return ToolResult("compare_symmetric", False, {}, str(e))

    # Legacy alias
    def compare_symmetric_regions(self, axis: str = "vertical") -> ToolResult:
        return self.compare_symmetric(axis)

    # ---------------------------------------------------------------- Tool 4

    def get_scale_profile(self, region: str = "full", bbox=None) -> ToolResult:
        """
        Tool 4: SURFACE vs DEEP — reads actual per-scale z_scale_maps.

        z_scale_maps[3] = F4 finest   (surface texture) → scratches, stains
        z_scale_maps[0] = F1 coarsest (structural)      → cracks, voids

        Previously this function faked scale by using different percentiles
        of the single anomaly_map, which made fine and coarse always correlated
        and the discrimination useless. Now reads the correct layers.
        """
        try:
            if self.z_scale_maps is None:
                return ToolResult("get_scale_profile", False, {},
                    "z_scale_maps not available — Stage 1 calibration must run first")

            mask = self._get_region_mask(region, bbox)
            if mask.sum() == 0:
                return ToolResult("get_scale_profile", False, {}, "Empty region")

            fine   = float(self.z_scale_maps[3][mask].max())   # F4 surface
            coarse = float(self.z_scale_maps[0][mask].max())   # F1 structural
            ratio  = fine / (abs(coarse) + 1e-8)

            if coarse < 0.5 and fine > 2.0:
                interp = "surface-only — RULES OUT crack, void, structural defects"
            elif abs(fine - coarse) < 0.5 and fine > 2.0:
                interp = "structural — RULES OUT contamination, stain, label defect"
            elif fine > 2.0:
                interp = "mixed — cannot eliminate by scale alone"
            else:
                interp = "weak signal — borderline case"

            return ToolResult("get_scale_profile", True, {
                "fine":                  round(fine, 2),
                "coarse":                round(coarse, 2),
                "fine_to_coarse_ratio":  round(ratio, 2),
                "interpretation":        interp,
                "fine_interpretation":   interpret_z(fine),
                "coarse_interpretation": interpret_z(coarse),
            })
        except Exception as e:
            return ToolResult("get_scale_profile", False, {}, str(e))

    # ---------------------------------------------------------------- Tool 5

    def retrieve_similar_cases(self, top_k: int = 3) -> ToolResult:
        """
        Tool 5: Retrieves similar past confirmed defect cases from RAGStore.

        Uses f3_embedding (L2-normalised [1024]-d) from state when available;
        falls back to 16-d surrogate feature vector otherwise.
        Only human-confirmed entries should influence classification confidence.
        """
        try:
            from .active_learning import get_active_learning_manager
            mgr      = get_active_learning_manager()
            category = self.state.get('category', 'unknown')

            deep_emb      = self.state.get('f3_embedding')
            deep_emb_list = deep_emb.tolist() if hasattr(deep_emb, 'tolist') else (
                list(deep_emb) if deep_emb is not None else None
            )

            if self.z_map is not None:
                z_proxy = float(self.z_map.max())
                area    = int((self.z_map > 2.0).sum())
            else:
                raw_max = float(self.anomaly_map.max())
                z_proxy = (raw_max - 0.50) / 0.20
                area    = int((self.anomaly_map > raw_max * 0.5).sum())

            aspect_ratio = 1.0
            try:
                sr = self.analyze_shape(z_threshold=2.0)
                if sr.success:
                    aspect_ratio = sr.data.get('aspect_ratio', 1.0)
            except Exception:
                pass

            if self.z_map is not None:
                left  = float(self.z_map[:, :self.W//2].mean())
                right = float(self.z_map[:, self.W//2:].mean())
            else:
                left  = float(self.anomaly_map[:, :self.W//2].mean())
                right = float(self.anomaly_map[:, self.W//2:].mean())
            asymmetry = (max(left, right) + 1e-8) / (min(left, right) + 1e-8)

            similar = mgr.retrieve_similar(
                category=category,
                z_score=z_proxy,
                area=area,
                aspect_ratio=aspect_ratio,
                asymmetry=asymmetry,
                top_k=top_k,
            )

            if not similar:
                return ToolResult("retrieve_similar_cases", True, {
                    "cases": [],
                    "n_total_in_store": len(mgr.rag),
                    "note": "No similar cases in RAG store yet.",
                })

            def quality(sim):
                if sim > 0.95: return "very high — near-identical defect"
                if sim > 0.85: return "high — closely related"
                if sim > 0.70: return "moderate — similar category"
                return "low — may not be relevant"

            cases_summary = [{
                "id":               c.get("id"),
                "defect_type":      c.get("defect_type"),
                "severity":         c.get("severity"),
                "confidence":       c.get("confidence"),
                "confirmed_by":     c.get("confirmed_by"),
                "similarity":       c.get("similarity"),
                "root_cause":       c.get("root_cause", ""),
                "human_answer":     c.get("human_answer", ""),
                "trust_level":      "human-confirmed" if c.get("confirmed_by") == "human"
                                    else "auto-accepted (lower trust)",
                "retrieval_quality": quality(c.get("similarity", 0)),
            } for c in similar]

            return ToolResult("retrieve_similar_cases", True, {
                "cases":            cases_summary,
                "n_retrieved":      len(cases_summary),
                "n_total_in_store": len(mgr.rag),
                "note": ("Only human-confirmed entries should influence your verdict. "
                         "Auto-accepted entries provide context only."),
            })
        except Exception as e:
            return ToolResult("retrieve_similar_cases", False, {}, str(e))

    # ---------------------------------------------------------------- dispatch

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool_map = {
            "query_region":              self.query_region,
            "analyze_shape":             self.analyze_shape,
            "compare_symmetric":         self.compare_symmetric,
            "compare_symmetric_regions": self.compare_symmetric_regions,
            "get_scale_profile":         self.get_scale_profile,
            "retrieve_similar_cases":    self.retrieve_similar_cases,
        }
        if tool_name not in tool_map:
            return ToolResult(tool_name, False, {}, f"Unknown tool: {tool_name}")
        return tool_map[tool_name](**kwargs)
