"""Traceprop-LLM: inline gradient provenance for transformer + LoRA training.

Public API::

    from traceprop.llm import LoRAGradientLogger, select_lora_linears
"""
from traceprop.llm.lora_logging import LoRAGradientLogger, select_lora_linears

__all__ = ["LoRAGradientLogger", "select_lora_linears"]
