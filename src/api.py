"""
FastAPI server for Symptom-Diagnosis-GPT inference.
Provides REST API endpoints for symptom-to-diagnosis prediction.
"""
import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
import tiktoken
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_model_config
from .model import SymptomDiagnosisGPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Symptom-Diagnosis-GPT API",
    description="AI-powered symptom analysis and diagnosis prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
config = None


class SymptomRequest(BaseModel):
    """Request model for symptom prediction."""
    symptoms: str = Field(..., description="Description of symptoms", min_length=1, max_length=500)
    max_length: Optional[int] = Field(50, description="Maximum tokens to generate", ge=1, le=200)
    temperature: Optional[float] = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)


class DiagnosisResponse(BaseModel):
    """Response model for diagnosis prediction."""
    diagnosis: str = Field(..., description="Predicted diagnosis")
    confidence: float = Field(..., description="Confidence score (0-1)")
    input_text: str = Field(..., description="Processed input text")
    generated_text: str = Field(..., description="Full generated text")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    tokenizer_loaded: bool
    device: str


def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    global model, tokenizer, config
    
    config = get_model_config()
    
    try:
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        logger.info("✅ Tokenizer loaded successfully")
        
        # Check if model file exists
        if not os.path.exists(config.model_save_path):
            logger.warning(f"⚠️  Model file not found: {config.model_save_path}")
            logger.info("Using untrained model for demonstration")
            
            # Create untrained model for demonstration
            config.vocab_size = tokenizer.n_vocab
            model = SymptomDiagnosisGPT(config)
            logger.info("✅ Untrained model created for demonstration")
            return
        
        # Load trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(config.model_save_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # New checkpoint format
            model_config = checkpoint.get("config", config)
            model = SymptomDiagnosisGPT(model_config)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"✅ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Legacy format - try to load directly
            config.vocab_size = tokenizer.n_vocab
            model = SymptomDiagnosisGPT(config)
            try:
                model.load_state_dict(checkpoint)
                logger.info("✅ Model loaded from legacy checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load legacy checkpoint: {e}")
                logger.info("Using untrained model")
        
        model.to(device)
        model.eval()
        logger.info(f"✅ Model loaded on device: {device}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.info("Using untrained model for demonstration")
        
        # Fallback to untrained model
        tokenizer = tiktoken.get_encoding("gpt2")
        config.vocab_size = tokenizer.n_vocab
        model = SymptomDiagnosisGPT(config)
        model.eval()


def preprocess_symptoms(symptoms: str) -> str:
    """Preprocess symptom text for model input."""
    # Basic preprocessing
    symptoms = symptoms.strip().lower()
    
    # Format as model expects
    formatted_text = f"Symptoms: {symptoms}\nDiagnosis:"
    
    return formatted_text


def postprocess_prediction(generated_text: str, input_text: str) -> Dict[str, str]:
    """Extract diagnosis from generated text."""
    # Remove input portion
    if input_text in generated_text:
        prediction = generated_text.replace(input_text, "").strip()
    else:
        prediction = generated_text.strip()
    
    # Extract diagnosis if it follows the expected format
    if "diagnosis:" in prediction.lower():
        diagnosis = prediction.lower().split("diagnosis:")[-1].strip()
    else:
        diagnosis = prediction.strip()
    
    # Clean up diagnosis
    diagnosis = diagnosis.split("\n")[0]  # Take first line
    diagnosis = diagnosis.split(".")[0]   # Take first sentence
    diagnosis = diagnosis.strip()
    
    if not diagnosis:
        diagnosis = "Unable to determine diagnosis from symptoms"
    
    return diagnosis


def calculate_confidence(logits: torch.Tensor) -> float:
    """Calculate confidence score from model logits."""
    with torch.no_grad():
        # Get probability distribution for next token
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        
        # Use max probability as confidence
        confidence = torch.max(probs).item()
        
        return min(confidence * 2, 1.0)  # Scale and cap at 1.0


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("🚀 Starting Symptom-Diagnosis-GPT API...")
    load_model_and_tokenizer()
    logger.info("✅ API startup complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        tokenizer_loaded=tokenizer is not None,
        device=str(next(model.parameters()).device) if model else "unknown"
    )


@app.post("/predict", response_model=DiagnosisResponse)
async def predict_diagnosis(request: SymptomRequest):
    """Predict diagnosis from symptoms."""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or tokenizer not loaded"
        )
    
    try:
        # Preprocess input
        input_text = preprocess_symptoms(request.symptoms)
        
        # Tokenize input
        input_tokens = tokenizer.encode(input_text)
        
        # Limit input length
        max_input_length = config.max_length - request.max_length
        if len(input_tokens) > max_input_length:
            input_tokens = input_tokens[:max_input_length]
        
        # Convert to tensor
        device = next(model.parameters()).device
        input_ids = torch.tensor([input_tokens], device=device)
        
        # Generate prediction
        with torch.no_grad():
            # Get logits for confidence calculation
            logits, _ = model(input_ids)
            confidence = calculate_confidence(logits)
            
            # Generate text
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_length,
                temperature=request.temperature
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        
        # Extract diagnosis
        diagnosis = postprocess_prediction(generated_text, input_text)
        
        return DiagnosisResponse(
            diagnosis=diagnosis,
            confidence=confidence,
            input_text=input_text,
            generated_text=generated_text
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        device = next(model.parameters()).device
        num_params = model.get_num_params()
        
        return {
            "model_type": "SymptomDiagnosisGPT",
            "parameters": num_params,
            "device": str(device),
            "config": {
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "n_embed": config.n_embed,
                "vocab_size": config.vocab_size,
                "max_length": config.max_length
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Symptom-Diagnosis-GPT API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict diagnosis from symptoms",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
            "/docs": "GET - API documentation"
        }
    }


# Legacy endpoint for backward compatibility
@app.post("/predict-legacy")
async def predict_legacy(symptoms: dict):
    """Legacy prediction endpoint."""
    text = symptoms.get("text", "")
    
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'text' field in request"
        )
    
    # Convert to new format
    request = SymptomRequest(symptoms=text)
    response = await predict_diagnosis(request)
    
    # Return in legacy format
    return {"diagnosis": response.diagnosis}


if __name__ == "__main__":
    import uvicorn
    
    config = get_model_config()
    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        reload=True
    )
