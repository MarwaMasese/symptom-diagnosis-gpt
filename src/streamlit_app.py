"""
Streamlit UI for Symptom-Diagnosis-GPT.
Provides a user-friendly web interface for symptom analysis.
"""
import streamlit as st
import requests
import json
import time
from typing import Dict, Any


# Page configuration
st.set_page_config(
    page_title="Symptom Diagnosis GPT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_diagnosis(symptoms: str, max_length: int = 50, temperature: float = 1.0) -> Dict[str, Any]:
    """Send prediction request to API."""
    try:
        payload = {
            "symptoms": symptoms,
            "max_length": max_length,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Please ensure the server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def get_model_info() -> Dict[str, Any]:
    """Get model information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get model info"}
    except:
        return {"error": "Cannot connect to API"}


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏥 Symptom Diagnosis GPT")
    st.markdown("### AI-Powered Symptom Analysis and Diagnosis Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # API health check
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.markdown("Please start the API server:")
        st.sidebar.code("cd src && python -m api", language="bash")
    
    # Model parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider(
        "Max Generation Length",
        min_value=10,
        max_value=200,
        value=50,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Sampling temperature (higher = more creative)"
    )
    
    # Model info
    if api_healthy:
        with st.sidebar.expander("Model Information"):
            model_info = get_model_info()
            if "error" not in model_info:
                st.json(model_info)
            else:
                st.error(model_info["error"])
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Describe Your Symptoms")
        
        # Example symptoms
        examples = [
            "Select an example...",
            "I have fever, cough, and sore throat",
            "I experience chest pain and shortness of breath",
            "I have persistent headache and nausea",
            "I feel fatigue, joint pain, and have a rash",
            "I have stomach pain, diarrhea, and fever"
        ]
        
        example_choice = st.selectbox("Or choose an example:", examples)
        
        # Symptom input
        if example_choice != examples[0]:
            default_text = example_choice
        else:
            default_text = ""
        
        symptoms_text = st.text_area(
            "Enter your symptoms:",
            value=default_text,
            height=120,
            placeholder="Describe your symptoms in detail... (e.g., 'I have fever, headache, and body aches')",
            help="Be as specific as possible about your symptoms, their duration, and severity."
        )
        
        # Predict button
        predict_button = st.button(
            "🔍 Analyze Symptoms",
            type="primary",
            disabled=not api_healthy or not symptoms_text.strip()
        )
        
        # Disclaimer
        st.warning(
            "⚠️ **Medical Disclaimer**: This tool is for educational purposes only. "
            "Always consult with a qualified healthcare professional for medical advice. "
            "Do not use this as a substitute for professional medical diagnosis or treatment."
        )
    
    with col2:
        st.subheader("Diagnosis Prediction")
        
        if predict_button and symptoms_text.strip():
            with st.spinner("Analyzing symptoms..."):
                result = predict_diagnosis(symptoms_text, max_length, temperature)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display results
                st.success("Analysis Complete!")
                
                # Main diagnosis
                st.markdown("### 🎯 Predicted Diagnosis")
                st.markdown(f"**{result['diagnosis']}**")
                
                # Confidence score
                confidence = result.get('confidence', 0.0)
                st.markdown("### 📊 Confidence Score")
                st.progress(confidence)
                st.markdown(f"Confidence: {confidence:.1%}")
                
                # Additional information
                with st.expander("View Details"):
                    st.markdown("**Processed Input:**")
                    st.code(result.get('input_text', ''), language="text")
                    
                    st.markdown("**Generated Text:**")
                    st.code(result.get('generated_text', ''), language="text")
        
        elif not api_healthy:
            st.info("Please start the API server to begin symptom analysis.")
        else:
            st.info("Enter your symptoms above to get a diagnosis prediction.")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔬 About This Tool")
        st.markdown(
            "This AI model uses a lightweight GPT-like transformer "
            "trained on symptom-diagnosis pairs to provide preliminary "
            "medical insights."
        )
    
    with col2:
        st.markdown("### 🚀 Distributed Training")
        st.markdown(
            "The model was trained using distributed computing with Ray, "
            "allowing for efficient training across multiple nodes and "
            "faster convergence."
        )
    
    with col3:
        st.markdown("### 💡 How to Use")
        st.markdown(
            "1. Enter your symptoms in detail\n"
            "2. Adjust generation parameters if needed\n"
            "3. Click 'Analyze Symptoms'\n"
            "4. Review the prediction and confidence score"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit, FastAPI, PyTorch, and Ray • "
        "Symptom-Diagnosis-GPT v1.0"
    )


def run_demo():
    """Run a demo version without API dependency."""
    st.title("🏥 Symptom Diagnosis GPT - Demo Mode")
    st.warning("Running in demo mode. API server not available.")
    
    symptoms = st.text_area("Enter symptoms:", placeholder="Describe your symptoms...")
    
    if st.button("Analyze (Demo)"):
        if symptoms:
            # Mock response
            import random
            mock_diagnoses = [
                "common cold", "flu", "migraine", "gastroenteritis",
                "allergic reaction", "muscle strain", "viral infection"
            ]
            
            diagnosis = random.choice(mock_diagnoses)
            confidence = random.uniform(0.6, 0.9)
            
            st.success(f"**Demo Diagnosis:** {diagnosis}")
            st.progress(confidence)
            st.info("This is a mock response. Start the API for real predictions.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Falling back to demo mode...")
        run_demo()