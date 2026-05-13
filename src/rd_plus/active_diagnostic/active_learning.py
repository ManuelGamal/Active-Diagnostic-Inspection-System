"""
Active Learning Extension for Active Diagnostic Pipeline
Stages 4-6: Confidence check → flag for human → feedback loop
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


class CalibrationTracker:
    """Tracks calibration updates from human feedback."""
    RETRAIN_THRESHOLD = 5  # Number of human samples needed per category


@dataclass
class HumanFeedback:
    """Record of human feedback on a case."""
    image_id: str
    original_verdict: Dict
    human_answer: str
    final_defect_type: str
    resolved_uncertainty: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ActiveLearningManager:
    """
    Manages the active learning loop:
    - Stage 4: Confidence check → auto-store OR flag for human
    - Stage 5: Generate targeted question for human
    - Stage 6: Update RAG + recalibrate
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        rag_store_path: str = "rag_store.json",
        calibration_path: str = "calibration.json",
        use_llm_questions: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.rag_store_path = Path(rag_store_path)
        self.calibration_path = Path(calibration_path)
        self.use_llm_questions = use_llm_questions
        
        # Load existing RAG store
        self.rag_store = self._load_json(self.rag_store_path, {})
        self.calibration = self._load_json(self.calibration_path, {})
        
        self.pending_reviews = []  # Cases needing human review
        self.rag = []  # In-memory RAG for retrieval
    
    def _load_json(self, path: Path, default):
        """Load JSON file or return default."""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default
    
    def _save_json(self, path: Path, data):
        """Save JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def stage4_confidence_check(self, verdict: Dict, state: Dict) -> str:
        """
        Stage 4: Check confidence and decide path.
        
        Returns: "auto_store" or "flag_for_human"
        """
        confidence = verdict.get("confidence", 0)
        unresolved = verdict.get("unresolved_uncertainty", "")
        
        # Flag if confidence is low OR there's unresolved uncertainty
        needs_review = (
            confidence < self.confidence_threshold or
            len(unresolved) > 10  # Non-trivial uncertainty
        )
        
        if needs_review:
            self.pending_reviews.append({
                "verdict": verdict,
                "state": state,
                "reason": f"confidence={confidence:.2f}, uncertainty={unresolved[:50]}"
            })
            return "flag_for_human"
        
        return "auto_store"
    
    def stage5_generate_question(self, verdict: Dict, state: Dict) -> str:
        """Stage 5: Generate targeted yes/no question using analysis data."""
        defect_type = verdict.get("defect_type", "unknown")
        unresolved = verdict.get("unresolved_uncertainty", "")
        location = verdict.get("location", "detected region")
        z = state.get('max_z', state.get('z_score', 0))
        ar = state.get('aspect_ratio', 1.0)
        category = state.get('category', 'unknown')

        if z >= 3.0 and ar > 2.5 and defect_type != 'unknown':
            q = f"The RD++ analysis shows an elongated defect (AR={ar:.1f}, z={z:.1f}σ) in this {category}. Is this a {defect_type}?"
        elif z >= 3.0 and defect_type != 'unknown':
            q = f"The analysis found a strong anomaly (z={z:.1f}σ) in this {category}. Does this look like a {defect_type}?"
        elif defect_type != 'unknown':
            q = f"The system detected an anomaly in this {category} (z={z:.1f}σ). The top diagnosis is {defect_type}. Is this correct?"
        else:
            q = f"The system detected an anomaly in this {category} (z={z:.1f}σ, AR={ar:.1f}). Can you identify the defect type?"

        if unresolved and unresolved.lower() not in ("none", "none.", "n/a", ""):
            q += f" Note: {unresolved}"
        return q
    
    def stage6_incorporate_feedback(
        self,
        image_id: str,
        human_answer: str,
        final_defect_type: str = None
    ):
        """
        Stage 6: Update RAG store and recalibrate based on feedback.
        """
        # Find the pending case
        case = None
        for p in self.pending_reviews:
            if p["state"].get("image_id") == image_id:
                case = p
                break
        
        if case is None:
            print(f"Warning: No pending case for {image_id}")
            return
        
        # Update RAG store
        image_hash = image_id  # Simplified - would use actual embedding
        
        if image_hash not in self.rag_store:
            self.rag_store[image_hash] = {}
        
        self.rag_store[image_hash].update({
            "defect_type": final_defect_type,
            "human_answer": human_answer,
            "confirmed_by": "human",
            "original_verdict": case["verdict"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Recalibrate if needed (simplified)
        category = case["state"].get("category", "unknown")
        if category not in self.calibration:
            self.calibration[category] = {"samples": []}
        
        # Add to calibration samples
        self.calibration[category]["samples"].append({
            "scalar_score": case["state"].get("scalar_score", 0),
            "confirmed_defect": final_defect_type != "normal"
        })
        
        # Save updates
        self._save_json(self.rag_store_path, self.rag_store)
        self._save_json(self.calibration_path, self.calibration)
        
        # Remove from pending
        self.pending_reviews = [p for p in self.pending_reviews 
                               if p["state"].get("image_id") != image_id]
        
        print(f"✓ Updated RAG + calibration for {image_id}")
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get list of cases needing human review."""
        return self.pending_reviews
    
    def auto_store_verdict(self, verdict: Dict, state: Dict):
        """Auto-store verdict when confidence is high enough."""
        image_id = state.get("image_id", "unknown")
        
        if image_id not in self.rag_store:
            self.rag_store[image_id] = {}
        
        self.rag_store[image_id].update({
            "defect_type": verdict.get("defect_type"),
            "confidence": verdict.get("confidence"),
            "severity": verdict.get("severity"),
            "auto_confirmed": True,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_json(self.rag_store_path, self.rag_store)
        print(f"✓ Auto-stored verdict for {image_id}")
    
    def get_rag_similar(self, defect_type: str = None, top_k: int = 3) -> List[Dict]:
        """Get similar past cases from RAG store."""
        if not self.rag_store:
            return []
        
        cases = list(self.rag_store.values())
        
        if defect_type:
            cases = [c for c in cases if c.get("defect_type") == defect_type]
        
        return cases[:top_k]

    def run(self, verdict: Dict, state: Dict) -> Dict:
        """
        Run the full active learning flow.
        Only adds to RAG when auto-accepted (high confidence, no human needed).
        Flagged cases wait for incorporate_feedback before entering RAG.
        """
        image_path = state.get('image_path', state.get('image_id', 'unknown'))
        
        action = self.stage4_confidence_check(verdict, state)
        
        question = None
        if action == "flag_for_human" and self.use_llm_questions:
            question = self.stage5_generate_question(verdict, state)
        
        import hashlib
        case_id = hashlib.md5(f"{image_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Only auto-store in RAG if high confidence (no human needed)
        if action == "auto_store":
            self.rag.append({
                'id': case_id,
                'defect_type': verdict.get('defect_type'),
                'confidence': verdict.get('confidence'),
                'severity': verdict.get('severity'),
                'category': state.get('category'),
                'max_z': state.get('max_z'),
                'area': state.get('area'),
                'aspect_ratio': state.get('aspect_ratio'),
                'asymmetry': state.get('asymmetry'),
                'confirmed_by': 'auto',
                'question': None,
                'human_answer': '',
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'case_id': case_id,
            'action': action.replace('flag_for_human', 'pending_review').replace('auto_store', 'auto_stored'),
            'question': question
        }
    
    def incorporate_feedback(self, case_id: str, human_answer: str, corrected_type: str = None) -> Dict:
        """Process human feedback and store in RAG."""
        verdict = corrected_type or 'unknown'
        state_info = {}
        
        # Build RAG entry from pending data
        rag_entry = {
            'id': case_id,
            'defect_type': corrected_type or verdict,
            'confidence': 0.92,
            'severity': 'medium',
            'confirmed_by': 'human',
            'human_answer': human_answer,
            'timestamp': datetime.now().isoformat()
        }
        self.rag.append(rag_entry)
        
        self.pending_reviews = [p for p in self.pending_reviews if p.get('case_id') != case_id]
        
        return {'status': 'feedback_received', 'case_id': case_id, 'rag_size': len(self.rag)}
    
    def retrieve_similar(self, category: str, z_score: float, area: int, 
                        aspect_ratio: float, asymmetry: float, top_k: int = 3) -> List[Dict]:
        """Retrieve similar cases from RAG."""
        if not self.rag:
            return []
        
        # Filter by category
        candidates = [c for c in self.rag if c.get('category') == category]
        
        # Score by similarity (simplified)
        scored = []
        for c in candidates:
            score = 1.0
            # Reduce score based on feature differences
            if c.get('max_z'):
                score -= abs(c.get('max_z', 0) - z_score) * 0.1
            if c.get('aspect_ratio'):
                score -= abs(c.get('aspect_ratio', 1) - aspect_ratio) * 0.2
            scored.append((score, c))
        
        scored.sort(reverse=True)
        return [c[1] for c in scored[:top_k] if c[0] > 0.3]
    
    def rag_summary(self) -> Dict:
        """Get summary of RAG store."""
        total = len(self.rag)
        by_category = {}
        human_confirmed = 0
        auto_confirmed = 0
        
        for case in self.rag:
            cat = case.get('category', 'unknown')
            by_category[cat] = by_category.get(cat, 0) + 1
            if case.get('confirmed_by') == 'human':
                human_confirmed += 1
            else:
                auto_confirmed += 1
        
        return {
            'total_cases': total,
            'by_category': by_category,
            'human_confirmed': human_confirmed,
            'auto_confirmed': auto_confirmed
        }
    
    def get_pending(self) -> List[Dict]:
        """Get pending reviews."""
        return self.pending_reviews
    
    def get_recalibrated_stats(self, category: str) -> Dict:
        """Get recalibrated stats for category."""
        # Simplified - just return learned stats if we have enough samples
        cat_cases = [c for c in self.rag if c.get('category') == category and c.get('confirmed_by') == 'human']
        n = len(cat_cases)
        
        if n >= CalibrationTracker.RETRAIN_THRESHOLD:
            z_scores = [c.get('max_z', 2.0) for c in cat_cases]
            return {
                'mu': float(sum(z_scores)) / n if n > 0 else 0.5,
                'sigma': 0.2,
                'n': n,
                'source': 'learned'
            }
        
        return None


# Convenience function for full pipeline
def run_full_pipeline_with_learning(
    state: Dict,
    verdict: Dict,
    chain_log: List,
    learning_manager: ActiveLearningManager = None
) -> Dict:
    """
    Run the full pipeline including active learning.
    
    Returns:
        dict with:
        - action: "auto_stored" or "pending_review"
        - verdict: original verdict
        - question: targeted question (if pending review)
    """
    if learning_manager is None:
        learning_manager = ActiveLearningManager()
    
    # Stage 4: Confidence check
    action = learning_manager.stage4_confidence_check(verdict, state)
    
    result = {
        "action": action,
        "verdict": verdict,
        "chain_log": chain_log
    }
    
    if action == "flag_for_human":
        # Stage 5: Generate targeted question
        question = learning_manager.stage5_generate_question(verdict, state)
        result["question"] = question
        result["pending_id"] = state.get("image_id", "unknown")
    else:
        # Auto-store
        learning_manager.auto_store_verdict(verdict, state)
    
    return result


# Example usage
if __name__ == "__main__":
    # Initialize
    manager = ActiveLearningManager(
        confidence_threshold=0.75,
        rag_store_path="data/rag_store.json",
        calibration_path="data/calibration.json"
    )
    
    # Simulate a verdict
    test_verdict = {
        "defect_type": "body_scratch",
        "confidence": 0.65,  # Below threshold - will flag
        "severity": "medium",
        "location": "center-body",
        "unresolved_uncertainty": "Cannot distinguish scratch from surface stain"
    }
    
    test_state = {
        "image_id": "bottle_001",
        "category": "bottle"
    }
    
    # Stage 4
    action = manager.stage4_confidence_check(test_verdict, test_state)
    print(f"Action: {action}")
    
    if action == "flag_for_human":
        # Stage 5
        question = manager.stage5_generate_question(test_verdict, test_state)
        print(f"Human question: {question}")
        
        # Stage 6 (simulated human answer)
        manager.stage6_incorporate_feedback(
            "bottle_001",
            human_answer="Yes, it's a scratch, not a stain",
            final_defect_type="body_scratch"
        )