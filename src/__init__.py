"""
Symptom-Diagnosis-GPT: A distributed transformer model for medical diagnosis prediction.

This package provides:
- Lightweight GPT-like transformer model
- Distributed training with Ray
- FastAPI inference server
- Streamlit web interface
"""

__version__ = "1.0.0"
__author__ = "Symptom-Diagnosis-GPT Team"

from .config import get_model_config, get_distributed_config
from .model import SymptomDiagnosisGPT
from .prepare_data import build_dataset, load_dataset

__all__ = [
    "get_model_config",
    "get_distributed_config", 
    "SymptomDiagnosisGPT",
    "build_dataset",
    "load_dataset"
]