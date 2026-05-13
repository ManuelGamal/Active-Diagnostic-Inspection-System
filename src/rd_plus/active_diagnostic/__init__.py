"""
Active Diagnostic Pipeline - LLM-driven spatial queries for industrial anomaly detection

This module implements the "Anomaly Detection as Active Bayesian Dialogue" paradigm:
- Stage 0: Pre-compute RD++ anomaly maps once
- Stage 1: Initial brief to LLM
- Stage 2: Query loop (LLM drives spatial queries)
- Stage 3: Synthesis (structured verdict)
- Stage 4: RAG update (store for future)

Key components:
- pipeline.py: Main orchestration
- tools.py: Tool execution layer (5 tools)
- system_prompts.py: Manufacturing knowledge prompts

Usage:
    from active_diagnostic import ActiveDiagnosticPipeline
    
    pipeline = ActiveDiagnosticPipeline(rd_model, llm_client)
    verdict, chain = pipeline.run(image, category)
"""

from .pipeline import ActiveDiagnosticPipeline, Verdict, DiagnosticState, ToolCall
from .tools import ToolExecutor, ToolResult
from .system_prompts import build_system_prompt, TOOL_SCHEMAS

__all__ = [
    "ActiveDiagnosticPipeline",
    "Verdict",
    "DiagnosticState", 
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "build_system_prompt",
    "TOOL_SCHEMAS"
]