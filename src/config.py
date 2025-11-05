"""
Configuration settings for Symptom-Diagnosis-GPT distributed training and inference.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT-like transformer model."""
    
    # Model architecture
    vocab_size: int = 8192  # Will be adjusted based on tokenizer
    max_length: int = 256   # Maximum sequence length
    n_layers: int = 4       # Number of transformer layers
    n_heads: int = 4        # Number of attention heads
    n_embed: int = 128      # Embedding dimension
    dropout: float = 0.1    # Dropout rate
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 100
    gradient_clip_val: float = 1.0
    
    # Distributed training
    num_workers: int = 2    # Number of distributed workers
    use_ray: bool = False   # Use Ray for distributed training (disabled by default)
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True if torch.cuda.is_available() else False
    
    # Paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_save_path: str = "data/processed/model.pt"
    tokenizer_save_path: str = "data/processed/tokenizer.json"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Ray configuration
    ray_address: Optional[str] = None  # "ray://head-node-ip:10001" for cluster
    num_cpus_per_worker: int = 2
    num_gpus_per_worker: float = 0.5 if torch.cuda.is_available() else 0
    
    # PyTorch DDP configuration (fallback)
    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Training configuration
    checkpoint_freq: int = 1  # Save checkpoint every N epochs
    sync_weights_freq: int = 50  # Sync weights every N steps
    

# Global configuration instances
model_config = ModelConfig()
distributed_config = DistributedConfig()


def get_model_config():
    """Get the global model configuration."""
    return model_config


def get_distributed_config():
    """Get the global distributed configuration."""
    return distributed_config


def update_config(**kwargs):
    """Update configuration parameters."""
    global model_config, distributed_config
    
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(distributed_config, key):
            setattr(distributed_config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")


# Legacy config for backward compatibility
config = {
    "block_size": model_config.max_length,
    "batch_size": model_config.batch_size,
    "n_layer": model_config.n_layers,
    "n_head": model_config.n_heads,
    "n_embd": model_config.n_embed,
    "max_iters": 5000,
    "lr": model_config.learning_rate,
    "dropout": model_config.dropout,
    "device": model_config.device
}
