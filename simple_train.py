"""
Simple training script without Ray dependency.
This script provides an easy way to train the model using PyTorch DDP or single-node training.
"""
import os
import sys
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_train():
    """Quick training without Ray dependency."""
    print("🏥 Symptom-Diagnosis-GPT - Simple Training")
    print("=" * 50)
    
    try:
        # Import after path setup
        from src.config import get_model_config, update_config
        from src.prepare_data import build_dataset, load_dataset
        from src.train import train_model
        
        print("📋 Step 1: Configuration")
        config = get_model_config()
        
        # Use smaller settings for quick demo
        update_config(
            max_epochs=5,
            batch_size=16,
            n_layers=4,
            n_heads=4,
            n_embed=128,
            use_ray=False,  # Ensure Ray is disabled
            num_workers=1   # Single node
        )
        
        print(f"✅ Configuration:")
        print(f"   - Epochs: {config.max_epochs}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Workers: {config.num_workers}")
        print(f"   - Ray disabled: {not config.use_ray}")
        
        print("\n📊 Step 2: Data Preparation")
        
        # Check if data exists
        train_data, val_data, test_data = load_dataset()
        
        if train_data is None:
            print("🔄 Creating synthetic dataset...")
            train_data, val_data, test_data = build_dataset(num_samples=500)
        else:
            print("✅ Using existing dataset")
        
        print(f"   - Training samples: {len(train_data['input_ids']) if train_data else 0}")
        
        print("\n🎯 Step 3: Training")
        print("🔄 Starting training (this may take a few minutes)...")
        
        # Train model
        model = train_model(use_real_data=True)
        
        print("✅ Training completed!")
        print(f"✅ Model saved to: {config.model_save_path}")
        
        print("\n🚀 Next Steps:")
        print("1. Start API server: python -m src.api")
        print("2. Start web UI: streamlit run src/streamlit_app.py")
        print("3. Test API: curl http://localhost:8000/health")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing missing dependencies:")
        print("   pip install torch tiktoken fastapi uvicorn streamlit")
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.exception("Training error details:")
        return False


def check_system():
    """Check system requirements."""
    print("🔍 System Check")
    print("-" * 30)
    
    # Check Python version
    import sys
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check dependencies
    required_packages = {
        "torch": "PyTorch",
        "tiktoken": "Tiktoken",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "streamlit": "Streamlit",
        "numpy": "NumPy",
        "pandas": "Pandas"
    }
    
    missing = []
    available = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            available.append(name)
        except ImportError:
            missing.append(name)
    
    print(f"\n📦 Available packages: {', '.join(available)}")
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n💡 Install missing packages:")
        print("   pip install torch tiktoken fastapi uvicorn streamlit numpy pandas")
        return False
    else:
        print("✅ All required packages available")
    
    # Check optional packages
    optional_packages = {
        "ray": "Ray (distributed training)"
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ Optional: {description}")
        except ImportError:
            print(f"⚠️  Optional: {description} not available")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Symptom-Diagnosis-GPT Training")
    parser.add_argument("--check", action="store_true", help="Check system requirements only")
    parser.add_argument("--train", action="store_true", help="Start training")
    
    args = parser.parse_args()
    
    if args.check:
        success = check_system()
        if success:
            print("\n🎉 System ready for training!")
        else:
            print("\n❌ System check failed")
        return
    
    if args.train:
        success = quick_train()
        if not success:
            print("\n💡 Try running system check first: python simple_train.py --check")
        return
    
    # Default: show help
    print("🏥 Simple Symptom-Diagnosis-GPT Training")
    print("=" * 50)
    print("Available commands:")
    print("  --check  : Check system requirements")
    print("  --train  : Start training")
    print("\nExample:")
    print("  python simple_train.py --check")
    print("  python simple_train.py --train")


if __name__ == "__main__":
    main()