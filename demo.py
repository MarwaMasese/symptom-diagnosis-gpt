"""
Quick demo script to test the Symptom-Diagnosis-GPT system.
This script demonstrates the complete pipeline from data preparation to inference.
"""
import os
import sys
import time
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_model_config, update_config
from src.prepare_data import build_dataset, load_dataset
from src.train import train_model
from src.model import SymptomDiagnosisGPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_complete_pipeline():
    """Demonstrate the complete training and inference pipeline."""
    
    print("🏥 Symptom-Diagnosis-GPT Demo")
    print("=" * 50)
    
    # 1. Configuration
    print("\n📋 Step 1: Configuration")
    config = get_model_config()
    
    # Use smaller settings for demo
    update_config(
        max_epochs=3,
        batch_size=16,
        n_layers=2,
        n_heads=2,
        n_embed=64
    )
    
    print(f"✅ Configuration updated:")
    print(f"   - Epochs: {config.max_epochs}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Model: {config.n_layers} layers, {config.n_heads} heads, {config.n_embed} dim")
    
    # 2. Data Preparation
    print("\n📊 Step 2: Data Preparation")
    
    # Check if data already exists
    train_data, val_data, test_data = load_dataset()
    
    if train_data is None:
        print("🔄 Creating synthetic dataset...")
        train_data, val_data, test_data = build_dataset(num_samples=200)  # Small for demo
    else:
        print("✅ Using existing dataset")
    
    print(f"   - Training samples: {len(train_data['input_ids'])}")
    print(f"   - Validation samples: {len(val_data['input_ids'])}")
    print(f"   - Test samples: {len(test_data['input_ids'])}")
    
    # 3. Training
    print("\n🎯 Step 3: Training")
    
    start_time = time.time()
    
    try:
        # Train with real data but small scale for demo
        model = train_model(use_real_data=True)
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("⚠️  Training failed, using dummy model for demo")
        
        # Create untrained model for demo
        model = SymptomDiagnosisGPT(config)
        print("✅ Demo model created")
    
    # 4. Model Info
    print("\n🔧 Step 4: Model Information")
    print(f"   - Parameters: {model.get_num_params():,}")
    print(f"   - Model size: {model.get_num_params() * 4 / 1024 / 1024:.2f} MB")
    print(f"   - Device: {next(model.parameters()).device}")
    
    # 5. Inference Demo
    print("\n🔍 Step 5: Inference Demo")
    
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Demo symptoms
        demo_symptoms = [
            "fever, cough, sore throat",
            "chest pain, shortness of breath", 
            "headache, nausea, dizziness"
        ]
        
        model.eval()
        
        for symptoms in demo_symptoms:
            input_text = f"Symptoms: {symptoms}\nDiagnosis:"
            print(f"\n📝 Input: {symptoms}")
            
            # Tokenize
            tokens = tokenizer.encode(input_text)
            tokens = tokens[:config.max_length//2]  # Leave room for generation
            
            # Generate (simplified for demo)
            import torch
            with torch.no_grad():
                input_ids = torch.tensor([tokens])
                generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0)
                
                # Decode
                generated_text = tokenizer.decode(generated[0].tolist())
                prediction = generated_text.replace(input_text, "").strip()
                if not prediction:
                    prediction = "[No clear diagnosis generated]"
                
                print(f"🎯 Prediction: {prediction}")
    
    except Exception as e:
        logger.error(f"Inference demo failed: {e}")
        print("⚠️  Inference demo failed (this is normal without proper training)")
    
    # 6. Next Steps
    print("\n🚀 Step 6: Next Steps")
    print("To use the complete system:")
    print("   1. API Server: python -m src.api")
    print("   2. Web UI: streamlit run src/streamlit_app.py")
    print("   3. Distributed Training: python -m src.train_distributed")
    
    print("\n✅ Demo completed successfully!")
    print("📚 See README.md for full documentation")


def quick_test():
    """Quick test to verify imports and basic functionality."""
    print("🧪 Quick System Test")
    print("-" * 30)
    
    try:
        # Test imports
        from src.config import get_model_config
        from src.model import SymptomDiagnosisGPT
        from src.prepare_data import SymptomDatasetBuilder
        
        print("✅ All imports successful")
        
        # Test configuration
        config = get_model_config()
        print(f"✅ Configuration loaded (device: {config.device})")
        
        # Test model creation
        model = SymptomDiagnosisGPT(config)
        print(f"✅ Model created ({model.get_num_params():,} parameters)")
        
        # Test data builder
        builder = SymptomDatasetBuilder(config)
        sample_data = builder.create_synthetic_dataset(num_samples=5)
        print(f"✅ Data generation works ({len(sample_data)} samples)")
        
        print("\n🎉 All systems operational!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Symptom-Diagnosis-GPT Demo")
    parser.add_argument("--quick-test", action="store_true", help="Run quick system test only")
    parser.add_argument("--full-demo", action="store_true", help="Run complete pipeline demo")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    elif args.full_demo:
        demo_complete_pipeline()
    else:
        print("🏥 Symptom-Diagnosis-GPT")
        print("Available options:")
        print("  --quick-test  : Quick system verification")
        print("  --full-demo   : Complete pipeline demonstration")
        print("\nFor full usage, see README.md")