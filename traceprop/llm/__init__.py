"""Traceprop-LLM: inline gradient provenance for transformer + LoRA training.

Public API::

    from traceprop.llm import LoRAGradientLogger, select_lora_linears
"""
from traceprop.llm.lora_logging import (
    LoRAGradientLogger,
    apply_kfac_precondition,
    select_lora_linears,
)

__all__ = ["LoRAGradientLogger", "select_lora_linears", "apply_kfac_precondition"]
