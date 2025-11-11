"""
Unified LabGPT Web Application

A comprehensive web interface that integrates all LabGPT pipelines:
- RAG indexing (papers + lab documents)
- Data generation (code repos + papers)
- Model training (Llama 3.1 8B with LoRA)
- Chat interface (with RAG toggle)
- Grant generation (section-by-section with citations)

This application wraps the existing labgpt_cli.py orchestrator with a
user-friendly web interface and background task execution.
"""

__version__ = "1.0.0"
