import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class RAGStore:
    def __init__(self, path: str = "rag_store.json"):
        self.path = Path(path)
        self.cases: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.cases = json.loads(self.path.read_text())
            except Exception:
                self.cases = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cases, indent=2))

    def add_case(
        self,
        category: str,
        z_score: float,
        area: int,
        aspect_ratio: float,
        asymmetry: float,
        defect_type: str,
        confidence: float,
        severity: str,
        confirmed_by: str,
        human_answer: str = "",
        root_cause: str = "",
        chain_log: Optional[List] = None,
        deep_embedding: Optional[List] = None,
        image_path: str = "",
    ) -> str:
        case_id = hashlib.md5(f"{category}{image_path}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        case = {
            "id": case_id,
            "category": category,
            "z_score": z_score,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "asymmetry": asymmetry,
            "defect_type": defect_type,
            "confidence": confidence,
            "severity": severity,
            "confirmed_by": confirmed_by,
            "human_answer": human_answer,
            "root_cause": root_cause,
            "chain_log": chain_log or [],
            "deep_embedding": deep_embedding,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
        }
        self.cases.append(case)
        self._save()
        return case_id


_instance: Optional[RAGStore] = None


def get_rag_store() -> RAGStore:
    global _instance
    if _instance is None:
        _instance = RAGStore()
    return _instance