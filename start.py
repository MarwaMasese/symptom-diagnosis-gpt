"""
Startup script for Symptom-Diagnosis-GPT.
This script helps users get the system running quickly.
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path


def run_command(command, description, cwd=None, background=False):
    """Run a command with error handling."""
    print(f"🔄 {description}...")
    
    try:
        if background:
            # Run in background
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✅ {description} started (PID: {process.pid})")
            return process
        else:
            # Run and wait
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {description} completed")
                return True
            else:
                print(f"❌ {description} failed:")
                print(result.stderr)
                return False
                
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "torch", "tiktoken", "fastapi", "uvicorn", 
        "streamlit", "numpy", "pandas"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("📦 Installing missing packages...")
        
        install_cmd = f"pip install {' '.join(missing)}"
        if run_command(install_cmd, "Installing packages"):
            print("✅ All packages installed")
            return True
        else:
            print("❌ Failed to install packages")
            return False
    else:
        print("✅ All dependencies available")
        return True


def setup_data():
    """Setup the dataset."""
    print("📊 Setting up dataset...")
    
    # Check if processed data exists
    processed_dir = Path("data/processed")
    
    if processed_dir.exists() and list(processed_dir.glob("*.pkl")):
        print("✅ Dataset already exists")
        return True
    
    # Create dataset
    python_cmd = sys.executable
    cmd = f"{python_cmd} -m src.prepare_data --num-samples 500"
    
    return run_command(cmd, "Creating dataset")


def train_model():
    """Train the model."""
    print("🎯 Training model...")
    
    # Check if model exists
    model_path = Path("data/processed/model.pt")
    
    if model_path.exists():
        print("✅ Trained model already exists")
        return True
    
    # Train model (quick training for demo)
    python_cmd = sys.executable
    cmd = f"{python_cmd} -m src.train --epochs 3 --batch-size 16"
    
    return run_command(cmd, "Training model")


def start_api_server():
    """Start the API server."""
    print("🚀 Starting API server...")
    
    python_cmd = sys.executable
    cmd = f"{python_cmd} -m uvicorn src.api:app --host 127.0.0.1 --port 8000"
    
    return run_command(cmd, "API server", background=True)


def start_web_ui():
    """Start the Streamlit web UI."""
    print("🌐 Starting web interface...")
    
    python_cmd = sys.executable
    cmd = f"{python_cmd} -m streamlit run src/streamlit_app.py --server.port 8501"
    
    return run_command(cmd, "Web interface", background=True)


def wait_for_service(url, timeout=30):
    """Wait for a service to become available."""
    import requests
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    
    return False


def main():
    """Main startup routine."""
    print("🏥 Symptom-Diagnosis-GPT Startup")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install requirements manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Step 2: Setup data
    if not setup_data():
        print("❌ Data setup failed. Try running manually:")
        print("   python -m src.prepare_data")
        return
    
    # Step 3: Train model (optional, quick training)
    train_model()  # Continue even if training fails
    
    # Step 4: Start services
    print("\n🚀 Starting services...")
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Failed to start API server")
        return
    
    # Wait for API to be ready
    print("⏳ Waiting for API server...")
    if wait_for_service("http://127.0.0.1:8000/health"):
        print("✅ API server is ready")
    else:
        print("⚠️ API server may not be fully ready")
    
    # Start web UI
    ui_process = start_web_ui()
    if not ui_process:
        print("❌ Failed to start web interface")
        return
    
    # Wait for UI to be ready
    print("⏳ Waiting for web interface...")
    time.sleep(5)  # Give Streamlit time to start
    
    # Open browser
    print("\n🌐 Opening web browser...")
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    print("\n✅ Startup complete!")
    print("\n📍 Service URLs:")
    print("   🌐 Web Interface: http://localhost:8501")
    print("   🔌 API Server: http://localhost:8000")
    print("   📚 API Docs: http://localhost:8000/docs")
    
    print("\n🛑 To stop services:")
    print("   Press Ctrl+C in their respective terminals")
    print("   Or run: pkill -f 'uvicorn|streamlit'")
    
    # Keep processes running
    print("\n⏳ Services running... Press Ctrl+C to stop monitoring")
    try:
        while True:
            time.sleep(10)
            # Check if processes are still running
            if api_process.poll() is not None:
                print("⚠️ API server stopped")
                break
            if ui_process.poll() is not None:
                print("⚠️ Web interface stopped")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        try:
            api_process.terminate()
            ui_process.terminate()
        except:
            pass


def quick_demo():
    """Run a quick demo without starting services."""
    print("🧪 Quick Demo Mode")
    print("-" * 30)
    
    # Just run the demo script
    python_cmd = sys.executable
    cmd = f"{python_cmd} demo.py --quick-test"
    
    run_command(cmd, "Running quick test")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Symptom-Diagnosis-GPT Startup")
    parser.add_argument("--demo", action="store_true", help="Run quick demo only")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    else:
        main()