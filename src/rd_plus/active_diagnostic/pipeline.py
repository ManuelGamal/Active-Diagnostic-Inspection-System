"""
Active Diagnostic Pipeline - Main Orchestration

Six-stage pipeline per specification:
  Stage 0 — RD++ forward pass (one time, cache everything)
  Stage 1 — Z-score calibration (calibrate_map + calibrate_scale_maps)
  Stage 2 — Heuristic pre-filter (cheap numpy, produces top-2 hypotheses)
  Stage 3 — LLM active query loop (max 5 turns)
  Stage 4 — Confidence check + validate_verdict
  Stage 5 — Targeted human question (if flagged)
  Stage 6 — RAG update
"""

import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tools import ToolExecutor, ToolResult
from .system_prompts import build_system_prompt, TOOL_SCHEMAS
from .calibration import calibrate_map, calibrate_scale_maps, CATEGORY_STATS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75   # spec §Stage 4
MAX_CONFIDENCE       = 0.92   # spec §Stage 3 — never exceed
HEURISTIC_CONFIDENCE_CAP = 0.60  # spec §Stage 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticState:
    """Full state dict after Stage 0 + Stage 1."""
    # Stage 0 outputs
    anomaly_map:    np.ndarray            # [H, W] raw 1 - cosine_sim
    scale_maps:     np.ndarray            # [4, H, W] per-scale before aggregation
    teacher_feats:  list                  # raw encoder tensors (for F3 embedding)
    category:       str
    image_path:     str
    ground_truth:   str                   # for eval only — never shown to LLM
    original_image: np.ndarray            # [H, W, 3] uint8
    bbox:           Optional[Tuple[int, int, int, int]] = None

    # Stage 1 outputs (populated by calibrate())
    z_map:              Optional[np.ndarray] = None   # [H, W]
    prob_map:           Optional[np.ndarray] = None   # [H, W]
    z_scale_maps:       Optional[np.ndarray] = None   # [4, H, W]
    peak_z:             float = 0.0
    peak_prob:          float = 0.0
    peak_location:      Tuple[int, int] = (0, 0)
    bbox_coverage_2s:   float = 0.0
    interpretation:     str = "unknown"
    f3_embedding:       Optional[np.ndarray] = None   # [1024] L2-normalised

    def to_tool_state(self) -> Dict:
        """Convert to the flat dict expected by ToolExecutor."""
        return {
            'anomaly_map':   self.anomaly_map,
            'z_map':         self.z_map,
            'z_scale_maps':  self.z_scale_maps,
            'prob_map':      self.prob_map,
            'bbox':          self.bbox,
            'category':      self.category,
            'original_image': self.original_image,
            'f3_embedding':  self.f3_embedding,
            'peak_z':        self.peak_z,
            'peak_location': self.peak_location,
        }


@dataclass
class ToolCall:
    tool_name:  str
    arguments:  Dict[str, Any]
    result:     Optional[ToolResult] = None
    turn:       int = 0


@dataclass
class Verdict:
    defect_type:            str
    confidence:             float
    severity:               str
    location:               str
    eliminated_types:       List[str]
    root_cause_candidates:  List[str]
    recommended_action:     str
    reasoning_summary:      str
    unresolved_uncertainty: str

    def to_dict(self) -> Dict:
        return {
            "defect_type":            self.defect_type,
            "confidence":             self.confidence,
            "severity":               self.severity,
            "location":               self.location,
            "eliminated_types":       self.eliminated_types,
            "root_cause_candidates":  self.root_cause_candidates,
            "recommended_action":     self.recommended_action,
            "reasoning_summary":      self.reasoning_summary,
            "unresolved_uncertainty": self.unresolved_uncertainty,
        }


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def _compute_f3_embedding(teacher_feats: list, z_map: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract L2-normalised [1024]-d F3 masked GAP embedding.
    teacher_feats[2] is F3 (index 2, medium-fine scale).
    Falls back to None if teacher_feats is empty (mock mode).
    """
    if not teacher_feats or len(teacher_feats) < 3:
        return None
    try:
        import torch
        feat = teacher_feats[2]  # F3, shape [1, C, h, w]
        if hasattr(feat, 'detach'):
            feat = feat.detach().cpu().numpy()
        feat = feat.squeeze(0)  # [C, h, w]

        # Mask: upsample z_map to feature resolution, threshold at 2σ
        import torch.nn.functional as F_nn
        H, W = z_map.shape
        h, w = feat.shape[1], feat.shape[2]
        z_tensor = torch.from_numpy(z_map).unsqueeze(0).unsqueeze(0).float()
        z_up = F_nn.interpolate(z_tensor, size=(h, w), mode='bilinear',
                                align_corners=False).squeeze().numpy()
        mask = z_up > 2.0
        if mask.sum() == 0:
            mask = np.ones((h, w), dtype=bool)

        # Masked global average pooling
        masked = feat[:, mask]  # [C, N]
        emb = masked.mean(axis=1)  # [C]

        # Pad or truncate to 1024
        if emb.shape[0] < 1024:
            emb = np.pad(emb, (0, 1024 - emb.shape[0]))
        else:
            emb = emb[:1024]

        # L2 normalise
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)
    except Exception:
        return None


def _run_stage1(state: 'DiagnosticState') -> 'DiagnosticState':
    """Stage 1: Z-score calibration."""
    stats = CATEGORY_STATS.get(state.category, CATEGORY_STATS['bottle'])

    z_map, prob_map, _ = calibrate_map(state.anomaly_map, state.category)
    z_scale_maps       = calibrate_scale_maps(state.scale_maps, stats)

    peak_idx         = np.unravel_index(z_map.argmax(), z_map.shape)
    peak_z           = float(z_map.max())
    peak_prob        = float(prob_map[peak_idx])
    peak_location    = (int(peak_idx[1]), int(peak_idx[0]))  # (x, y)

    # Fraction of bbox pixels above 2σ
    bbox_coverage_2s = 0.0
    if state.bbox is not None:
        x0, y0, x1, y1 = state.bbox
        bbox_z = z_map[y0:y1, x0:x1]
        if bbox_z.size > 0:
            bbox_coverage_2s = float((bbox_z > 2.0).mean())

    if peak_z >= 3.0:
        interpretation = "clear defect"
    elif peak_z >= 2.0:
        interpretation = "suspicious"
    elif peak_z >= 1.0:
        interpretation = "borderline"
    else:
        interpretation = "normal"

    f3_embedding = _compute_f3_embedding(state.teacher_feats, z_map)

    state.z_map            = z_map
    state.prob_map         = prob_map
    state.z_scale_maps     = z_scale_maps
    state.peak_z           = peak_z
    state.peak_prob        = peak_prob
    state.peak_location    = peak_location
    state.bbox_coverage_2s = bbox_coverage_2s
    state.interpretation   = interpretation
    state.f3_embedding     = f3_embedding
    return state


# ---------------------------------------------------------------------------
# Stage 2 — Heuristic pre-filter
# ---------------------------------------------------------------------------

def _get_all_defect_types(category: str) -> List[str]:
    catalogue = {
        'bottle':   ['rim_crack', 'rim_chip', 'body_scratch', 'body_crack',
                     'contamination', 'void_bubble', 'label_defect'],
        'capsule':  ['crack', 'faulty_imprint', 'poke', 'scratch',
                     'squeeze_damage', 'color_variation'],
        'carpet':   ['cut', 'hole', 'color_variation', 'thread_damage', 'contamination'],
        'hazelnut': ['crack', 'hole', 'print_defect', 'contamination', 'scratch'],
        'leather':  ['cut', 'fold', 'poke', 'color_variation', 'scratch'],
        'pill':     ['crack', 'broken', 'contamination', 'color_change', 'scratch'],
    }
    return catalogue.get(category, ['unknown'])


def heuristic_prefilter(state: 'DiagnosticState') -> Dict:
    """
    Stage 2: cheap numpy pre-filter. Must run BEFORE the LLM call.

    Returns:
        hypotheses    top-2 remaining defect types
        ruled_out     eliminated types with reasons
        fine_z        max fine-scale z in bbox
        coarse_z      max coarse-scale z in bbox
        aspect_ratio  AR of binary z_map > 2.0 mask
        n_components  number of connected components
    """
    from skimage.measure import label as sk_label, regionprops

    z_map        = state.z_map
    z_scale_maps = state.z_scale_maps
    bbox         = state.bbox
    category     = state.category

    # --- build bbox mask ------------------------------------------------
    H, W = z_map.shape
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
    else:
        mask = np.ones((H, W), dtype=bool)

    # --- scale signals --------------------------------------------------
    fine_masked = z_scale_maps[3][mask]
    coarse_masked = z_scale_maps[0][mask]
    fine   = float(fine_masked.max()) if fine_masked.size > 0 else 0.0
    coarse = float(coarse_masked.max()) if coarse_masked.size > 0 else 0.0

    # --- shape signals on z_map > 2.0 mask (NOT bbox mask) -------------
    binary  = z_map > 2.0
    labeled = sk_label(binary)
    regions = regionprops(labeled)

    ar = 1.0
    n_components = len(regions)
    if regions:
        largest = max(regions, key=lambda r: r.area)
        minor = largest.axis_minor_length
        major = largest.axis_major_length
        ar = major / minor if minor > 0 else 1.0

    # --- elimination rules ----------------------------------------------
    hypotheses = _get_all_defect_types(category)
    ruled_out  = []

    # Rule 1: scale profile
    if fine > 2.0 and coarse < 0.5:
        structural = ["crack", "void", "void_bubble", "chip", "deformation",
                      "body_crack", "rim_crack"]
        ruled_out.extend(structural)
    elif abs(fine - coarse) < 0.5 and fine > 2.0:
        surface_only = ["contamination", "color_stain", "color_variation",
                        "label_defect", "faulty_imprint", "print_defect"]
        ruled_out.extend(surface_only)

    # Rule 2: shape
    if ar > 2.5:
        compact = ["contamination", "hole", "poke", "void", "void_bubble"]
        ruled_out.extend(compact)
    elif ar < 1.3:
        elongated = ["crack", "scratch", "cut", "fold", "body_scratch", "rim_crack"]
        ruled_out.extend(elongated)

    # Rule 3: multiple components
    if n_components > 3:
        localized = ["crack", "scratch", "hole", "poke", "rim_crack", "body_scratch"]
        ruled_out.extend(localized)

    remaining = [h for h in hypotheses if h not in ruled_out]
    top2      = remaining[:2] if len(remaining) >= 2 else remaining

    return {
        "hypotheses":   top2,
        "ruled_out":    list(set(ruled_out)),
        "fine_z":       round(fine, 2),
        "coarse_z":     round(coarse, 2),
        "aspect_ratio": round(ar, 2),
        "n_components": n_components,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Confidence check + validate_verdict
# ---------------------------------------------------------------------------

def validate_verdict(verdict: Dict, chain_log: List[ToolCall]) -> Dict:
    """
    Cross-check verdict against tool outputs. Override clearly contradicted verdicts.
    Any overridden verdict is forced below 0.65 and flagged for human review.
    Only applies overrides when there's actual tool evidence in chain_log.
    """
    shape_result = next(
        (tc.result.data for tc in chain_log if tc.tool_name == "analyze_shape"
         and tc.result and tc.result.success), {}
    )
    scale_result = next(
        (tc.result.data for tc in chain_log if tc.tool_name == "get_scale_profile"
         and tc.result and tc.result.success), {}
    )

    has_tool_evidence = bool(shape_result or scale_result)
    
    defect = verdict.get("defect_type", "").lower()
    ar     = shape_result.get("aspect_ratio", 1.0)
    n_comp = shape_result.get("n_components", 1)
    fine   = scale_result.get("fine",   0.0)
    coarse = scale_result.get("coarse", 0.0)

    overridden = False

    # Only override when there's actual tool evidence
    if has_tool_evidence:
        # Contamination requires circular + diffuse + surface-only
        if defect == "contamination":
            if ar > 2.0:
                verdict["defect_type"]           = "scratch"
                verdict["confidence"]            = 0.60
                verdict["override_reason"]       = f"AR={ar:.1f}>2.0 rules out contamination"
                overridden = True
            elif n_comp == 1 and fine < 2.0:
                verdict["defect_type"]           = "unknown"
                verdict["confidence"]            = 0.45
                verdict["override_reason"]       = "single component + weak signal rules out contamination"
                overridden = True

        # Color/stain requires diffuse area
        if defect in ("color", "color_stain", "color_variation"):
            if ar > 3.0:
                verdict["defect_type"]     = "cut"
                verdict["confidence"]      = 0.60
                verdict["override_reason"] = f"AR={ar:.1f}>3.0 rules out color stain (too linear)"
                overridden = True

        # Crack requires both scales elevated
        if defect == "crack":
            if coarse < 0.5:
                verdict["defect_type"]     = "scratch"
                verdict["confidence"]      = 0.60
                verdict["override_reason"] = f"coarse={coarse:.2f}<0.5 rules out crack (surface only)"
                overridden = True

    if overridden:
        verdict["confidence"]             = min(verdict["confidence"], HEURISTIC_CONFIDENCE_CAP)
        verdict["unresolved_uncertainty"] = (
            f"Verdict overridden by tool-evidence validator: "
            f"{verdict.get('override_reason', '')}. Human review required."
        )

    # Enforce global confidence cap
    verdict["confidence"] = min(float(verdict.get("confidence", 0.5)), MAX_CONFIDENCE)
    return verdict


def check_confidence(verdict: Dict) -> str:
    """
    Stage 4: decide auto_accept vs flag_for_review.
    Returns 'auto_accept' or 'flag_for_review'.
    """
    confidence  = min(float(verdict.get("confidence", 0.0)), MAX_CONFIDENCE)
    uncertainty = verdict.get("unresolved_uncertainty", "")
    has_uncertainty = bool(
        uncertainty and
        uncertainty.lower() not in ("none", "none.", "n/a", "")
    )
    if confidence >= CONFIDENCE_THRESHOLD and not has_uncertainty:
        return "auto_accept"
    return "flag_for_review"


# ---------------------------------------------------------------------------
# Stage 5 — Targeted human question
# ---------------------------------------------------------------------------

QUESTION_PROMPT_TEMPLATE = """
You diagnosed {defect_type} with confidence {confidence:.0%}.
Your unresolved uncertainty: {uncertainty}

The two remaining candidates were: {candidates}

The anomaly peak is at {peak_location} with z-score {peak_z:.1f}σ.
Shape: AR={ar:.1f}, components={n_components}
Scale: fine={fine_z:.1f}σ, coarse={coarse_z:.1f}σ

Generate ONE yes/no question that directly distinguishes between the two candidates.
Reference the specific image location.
The question must be answerable by a non-expert looking at the image.

Output ONLY the question. No preamble.
"""


def generate_human_question(verdict: Dict, state: 'DiagnosticState',
                             chain_log: List[ToolCall]) -> str:
    """Stage 5: generate one targeted yes/no question."""
    shape_result = next(
        (tc.result.data for tc in chain_log if tc.tool_name == "analyze_shape"
         and tc.result and tc.result.success), {}
    )
    scale_result = next(
        (tc.result.data for tc in chain_log if tc.tool_name == "get_scale_profile"
         and tc.result and tc.result.success), {}
    )

    candidates = verdict.get("root_cause_candidates", [])
    prompt = QUESTION_PROMPT_TEMPLATE.format(
        defect_type   = verdict.get("defect_type", "unknown"),
        confidence    = verdict.get("confidence", 0.5),
        uncertainty   = verdict.get("unresolved_uncertainty", ""),
        candidates    = ", ".join(candidates[:2]) if candidates else "unknown",
        peak_location = state.peak_location,
        peak_z        = state.peak_z,
        ar            = shape_result.get("aspect_ratio", 1.0),
        n_components  = shape_result.get("n_components", 1),
        fine_z        = scale_result.get("fine", 0.0),
        coarse_z      = scale_result.get("coarse", 0.0),
    )
    return prompt.strip()


# ---------------------------------------------------------------------------
# Stage 6 — RAG update
# ---------------------------------------------------------------------------

def rag_update(verdict: Dict, state: 'DiagnosticState',
               chain_log: List[ToolCall],
               confirmed_by: str = "auto",
               human_answer: str = "") -> Optional[str]:
    """
    Stage 6: store confirmed verdict in RAG.

    Per spec:
      - Auto-accepted entries stored with confirmed_by='auto' — cannot boost future confidence above 0.75
      - Overridden verdicts NOT stored until human confirms corrected type
      - F3 embedding used when available
    """
    # Do not store overridden verdicts until human confirms
    if "override_reason" in verdict:
        return None

    try:
        from .rag_store import get_rag_store
        rag = get_rag_store()

        shape_result = next(
            (tc.result.data for tc in chain_log if tc.tool_name == "analyze_shape"
             and tc.result and tc.result.success), {}
        )
        z_score      = state.peak_z
        area         = int((state.z_map > 2.0).sum()) if state.z_map is not None else 0
        aspect_ratio = shape_result.get("aspect_ratio", 1.0)

        if state.z_map is not None:
            left  = float(state.z_map[:, :state.z_map.shape[1]//2].mean())
            right = float(state.z_map[:, state.z_map.shape[1]//2:].mean())
            asymmetry = (max(left, right) + 1e-8) / (min(left, right) + 1e-8)
        else:
            asymmetry = 1.0

        deep_emb = (state.f3_embedding.tolist()
                    if state.f3_embedding is not None else None)

        case_id = rag.add_case(
            category     = state.category,
            z_score      = z_score,
            area         = area,
            aspect_ratio = aspect_ratio,
            asymmetry    = asymmetry,
            defect_type  = verdict.get("defect_type", "unknown"),
            confidence   = verdict.get("confidence", 0.5),
            severity     = verdict.get("severity", "medium"),
            confirmed_by = confirmed_by,
            human_answer = human_answer,
            root_cause   = ", ".join(verdict.get("root_cause_candidates", [])),
            chain_log    = [{"tool": tc.tool_name, "result": tc.result.data if tc.result else {}}
                            for tc in chain_log],
            deep_embedding = deep_emb,
            image_path   = state.image_path,
        )
        return case_id
    except Exception as e:
        print(f"[RAG update failed] {e}")
        return None


def incorporate_human_answer(answer: str, verdict: Dict, state: 'DiagnosticState',
                              chain_log: List[ToolCall]) -> Tuple[Dict, Optional[str]]:
    """Stage 5 answer → Stage 6 RAG store."""
    candidates = verdict.get("root_cause_candidates", [])
    if answer.lower() in ("yes", "y", "true"):
        final_type = verdict["defect_type"]
    else:
        final_type = candidates[1] if len(candidates) > 1 else "unknown"

    verdict = dict(verdict)
    verdict["defect_type"]  = final_type
    verdict["confidence"]   = MAX_CONFIDENCE   # human confirmed
    verdict["confirmed_by"] = "human"

    case_id = rag_update(verdict, state, chain_log,
                          confirmed_by="human", human_answer=answer)
    return verdict, case_id


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ActiveDiagnosticPipeline:
    """
    Full six-stage active diagnostic pipeline.

    Usage:
        pipeline = ActiveDiagnosticPipeline(rd_model, llm_client)
        result = pipeline.run(image, category)
    """

    def __init__(self, rd_model=None, clip_model=None, llm_client=None,
                 max_turns: int = 5):
        self.rd_model  = rd_model
        self.llm_client = llm_client
        self.max_turns  = max_turns
        self._use_mock  = (rd_model is None or llm_client is None)

    # ---- Stage 0 -----------------------------------------------------------

    def _precompute(self, image: np.ndarray, category: str) -> DiagnosticState:
        if self._use_mock or self.rd_model is None:
            return self._mock_precompute(image, category)
        try:
            result = self.rd_model.forward_full(image)
            H, W = image.shape[:2]
            scale_dict = result.get('scale_maps', {})
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
            bbox = result.get('bbox', None)
            return DiagnosticState(
                anomaly_map=result.get('anomaly_map', np.zeros((H, W))),
                scale_maps=scale_maps,
                teacher_feats=result.get('teacher_feats', []),
                category=category,
                image_path=result.get('image_path', ''),
                ground_truth=result.get('ground_truth', ''),
                original_image=image,
                bbox=bbox,
            )
        except Exception as e:
            print(f"[Stage 0 real precompute failed: {e}] falling back to mock")
            return self._mock_precompute(image, category)

    def _mock_precompute(self, image: np.ndarray, category: str) -> DiagnosticState:
        H, W = image.shape[:2]
        anomaly_map = np.random.rand(H, W) * 0.3
        cy, cx = H // 2 + np.random.randint(-20, 20), W // 2 + np.random.randint(-20, 20)
        for i in range(H):
            for j in range(W):
                dist = np.sqrt((i - cy)**2 + 0.3 * (j - cx)**2)
                if dist < 40:
                    anomaly_map[i, j] = 0.5 + 0.5 * (1 - dist / 40)
        anomaly_map = np.clip(anomaly_map + np.random.rand(H, W) * 0.1, 0, 2)

        # Simulate 4-scale maps (in real impl these come from the model)
        scale_maps = np.stack([
            anomaly_map * (0.5 + 0.1 * k) + np.random.rand(H, W) * 0.05
            for k in range(4)
        ])  # [4, H, W]

        bbox = (cx - 30, cy - 30, cx + 30, cy + 30)

        return DiagnosticState(
            anomaly_map   = anomaly_map,
            scale_maps    = scale_maps,
            teacher_feats = [],   # empty in mock — f3_embedding will be None
            category      = category,
            image_path    = "",
            ground_truth  = "",
            original_image = image,
            bbox          = bbox,
        )

    # ---- Stage 1 -----------------------------------------------------------

    def _calibrate(self, state: DiagnosticState) -> DiagnosticState:
        return _run_stage1(state)

    # ---- Stage 2 -----------------------------------------------------------

    def _prefilter(self, state: DiagnosticState) -> Dict:
        return heuristic_prefilter(state)

    # ---- Stage 3 -----------------------------------------------------------

    def _build_initial_brief(self, state: DiagnosticState, prefilter: Dict) -> str:
        hyp_str     = ", ".join(prefilter["hypotheses"]) if prefilter["hypotheses"] else "unknown"
        ruled_str   = ", ".join(prefilter["ruled_out"])  if prefilter["ruled_out"]  else "none"
        bbox_str    = str(state.bbox) if state.bbox else "not detected"
        return (
            f"## Initial Anomaly Analysis\n\n"
            f"Category: {state.category}\n"
            f"Peak z-score: {state.peak_z:.1f}σ  ({state.interpretation})\n"
            f"Peak location: {state.peak_location}\n"
            f"Defect bbox: {bbox_str}\n\n"
            f"Pre-filter result:\n"
            f"  fine_z={prefilter['fine_z']}, coarse_z={prefilter['coarse_z']}, "
            f"AR={prefilter['aspect_ratio']}, n_components={prefilter['n_components']}\n"
            f"  Ruled out: {ruled_str}\n"
            f"  Remaining hypotheses: {hyp_str}\n\n"
            f"Start from these two hypotheses only. "
            f"Call get_scale_profile FIRST, then analyze_shape SECOND."
        )

    def _execute_tool(self, tool_name: str, args: Dict,
                      state: DiagnosticState) -> ToolResult:
        executor = ToolExecutor(state.to_tool_state())
        return executor.execute(tool_name, **args)

    def _parse_verdict(self, json_str: str) -> Optional[Dict]:
        import re
        # strip any markdown fences
        json_str = re.sub(r'```(?:json)?', '', json_str).strip()
        try:
            parsed = json.loads(json_str)
            if "defect_type" in parsed:
                return parsed
        except Exception:
            pass
        m = re.search(r'\{[\s\S]*"defect_type"[\s\S]*\}', json_str)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None

    # ---- Main run ----------------------------------------------------------

    def run(self, image: np.ndarray, category: str) -> Dict:
        """
        Run all six pipeline stages.

        Returns dict with: verdict, chain_log, action, question, case_id.
        """
        print(f"\n[Stage 0] Pre-computing for {category}...")
        state = self._precompute(image, category)

        # ---- Stage 1 -------------------------------------------------------
        print("[Stage 1] Z-score calibration...")
        state = self._calibrate(state)
        print(f"  peak_z={state.peak_z:.2f} ({state.interpretation})")

        # Early exit: no LLM needed if peak_z < 2.0
        if state.peak_z < 2.0:
            print("[Stage 1] peak_z < 2.0 — no defect detected, skipping LLM")
            no_defect: Dict = {
                "defect_type": "normal", "confidence": 0.90,
                "severity": "low", "location": "N/A",
                "eliminated_types": [], "root_cause_candidates": [],
                "recommended_action": "No action required",
                "reasoning_summary": f"Peak z={state.peak_z:.2f} < 2.0σ — below detection threshold",
                "unresolved_uncertainty": "",
            }
            case_id = rag_update(no_defect, state, [], confirmed_by="auto")
            return {"verdict": no_defect, "chain_log": [], "action": "auto_accept",
                    "question": None, "case_id": case_id}

        # ---- Stage 2 -------------------------------------------------------
        print("[Stage 2] Heuristic pre-filter...")
        prefilter = self._prefilter(state)
        print(f"  top-2: {prefilter['hypotheses']}  ruled_out: {prefilter['ruled_out']}")

        # ---- Stage 3 -------------------------------------------------------
        print(f"[Stage 3] LLM query loop (max {self.max_turns} turns)...")
        system_prompt   = build_system_prompt(category)
        initial_brief   = self._build_initial_brief(state, prefilter)
        messages        = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": initial_brief},
        ]
        chain_log: List[ToolCall] = []
        raw_verdict: Optional[Dict] = None

        for turn in range(self.max_turns):
            print(f"\n  -- Turn {turn + 1} --")
            response_text, tool_call = self._simulate_llm_response(messages, turn, prefilter)

            if tool_call is None:
                messages.append({"role": "assistant", "content": response_text})
                raw_verdict = self._parse_verdict(response_text)
                break

            tool_name = tool_call["name"]
            args      = tool_call.get("arguments", {})
            print(f"  Executing {tool_name}({args})")
            result = self._execute_tool(tool_name, args, state)
            print(f"  → {result.data}")

            tc = ToolCall(tool_name=tool_name, arguments=args, result=result, turn=turn)
            chain_log.append(tc)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool",      "content": json.dumps(result.data)})
            messages.append({"role": "user",      "content":
                             "Continue your analysis. When ready, output the verdict JSON."})

        # ---- Stage 4 -------------------------------------------------------
        if raw_verdict is None:
            raw_verdict = {
                "defect_type": prefilter["hypotheses"][0] if prefilter["hypotheses"] else "unknown",
                "confidence": HEURISTIC_CONFIDENCE_CAP,
                "severity": "medium",
                "location": str(state.peak_location),
                "eliminated_types": prefilter["ruled_out"],
                "root_cause_candidates": prefilter["hypotheses"],
                "recommended_action": "Manual review required — query loop did not converge",
                "reasoning_summary": "LLM did not produce a verdict within the turn limit",
                "unresolved_uncertainty": "Could not converge to a verdict",
            }

        print("\n[Stage 4] Validate verdict + confidence check...")
        raw_verdict  = validate_verdict(raw_verdict, chain_log)
        action       = check_confidence(raw_verdict)
        print(f"  action={action}  confidence={raw_verdict.get('confidence', 0):.2f}")

        # ---- Stage 5 -------------------------------------------------------
        question: Optional[str] = None
        if action == "flag_for_review":
            print("[Stage 5] Generating targeted human question...")
            question = generate_human_question(raw_verdict, state, chain_log)
            print(f"  Q: {question[:80]}...")

        # ---- Stage 6 -------------------------------------------------------
        print("[Stage 6] RAG update...")
        confirmed_by = "auto" if action == "auto_accept" else "pending"
        case_id      = rag_update(raw_verdict, state, chain_log, confirmed_by=confirmed_by)
        print(f"  stored case_id={case_id}")

        return {
            "verdict":   raw_verdict,
            "chain_log": chain_log,
            "action":    action,
            "question":  question,
            "case_id":   case_id,
        }

    def _simulate_llm_response(self, messages, turn, prefilter):
        """Mock LLM — follows spec tool call order: scale_profile first, shape second."""
        ordered_tools = [
            ("get_scale_profile", {"region": "full"}),
            ("analyze_shape",     {"z_threshold": 2.0}),
            ("compare_symmetric", {"axis": "vertical"}),
            ("query_region",      {"region": "bbox_interior", "scale": "fine", "aggregate": "max"}),
            ("retrieve_similar_cases", {"top_k": 3}),
        ]
        if turn < len(ordered_tools):
            name, args = ordered_tools[turn]
            return (f"Calling {name} to gather evidence.", {"name": name, "arguments": args})

        hyp = prefilter.get("hypotheses", ["unknown"])
        verdict = {
            "defect_type":            hyp[0] if hyp else "unknown",
            "confidence":             0.82,
            "severity":               "medium",
            "location":               "detected region",
            "eliminated_types":       prefilter.get("ruled_out", []),
            "root_cause_candidates":  hyp[:2],
            "recommended_action":     "Inspect production line",
            "reasoning_summary":      "Scale profile + shape analysis converged on verdict.",
            "unresolved_uncertainty": "",
        }
        return json.dumps(verdict), None


def demo():
    image    = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pipeline = ActiveDiagnosticPipeline()
    result   = pipeline.run(image, "bottle")
    print("\n=== RESULT ===")
    print(json.dumps(result["verdict"], indent=2))
    print(f"action={result['action']}  case_id={result['case_id']}")
    return result


if __name__ == "__main__":
    demo()
