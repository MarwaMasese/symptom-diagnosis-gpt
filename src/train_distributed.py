"""
Distributed training script for Symptom-Diagnosis-GPT using Ray.
Supports distributed training across multiple nodes/workers with fallback to PyTorch DDP.
"""
import os
import sys
import time
import json
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Ray imports (with fallback handling)
RAY_AVAILABLE = False
try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
    print("✅ Ray available for distributed training")
except ImportError:
    print("⚠️  Ray not available, using PyTorch DDP or single-node training")

# PyTorch distributed imports
try:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_DDP_AVAILABLE = True
    print("✅ PyTorch DDP available")
except ImportError:
    TORCH_DDP_AVAILABLE = False
    print("⚠️  PyTorch DDP not available")

# Local imports
from .config import get_model_config, get_distributed_config
from .model import SymptomDiagnosisGPT
from .prepare_data import load_dataset, build_dataset


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymptomDataset(Dataset):
    """PyTorch Dataset for symptom-diagnosis data."""
    
    def __init__(self, data: Dict[str, Any]):
        self.input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }


def create_data_loaders(train_data, val_data, config, rank=0, world_size=1):
    """Create distributed data loaders."""
    # Create datasets
    train_dataset = SymptomDataset(train_data)
    val_dataset = SymptomDataset(val_data)
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def compute_metrics(model, data_loader, device):
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, loss = model(input_ids, labels)
            
            total_loss += loss.item()
            
            # Compute accuracy (next token prediction)
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != 0)  # Ignore padding
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {"loss": avg_loss, "accuracy": accuracy}


def train_epoch(model, train_loader, optimizer, scheduler, device, config, epoch, rank=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    step = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        step += 1
        
        # Logging
        if batch_idx % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
            
            if rank == 0:  # Only log from main process
                logger.info(f"Epoch {epoch}, Step {batch_idx}: loss={avg_loss:.4f}, lr={lr:.2e}")
    
    return total_loss / len(train_loader)


def ray_train_func(config_dict: Dict[str, Any]):
    """Training function for Ray distributed training."""
    # Get configurations
    model_config = config_dict["model_config"]
    distributed_config = config_dict["distributed_config"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, val_data, _ = load_dataset()
    if train_data is None:
        logger.info("No processed data found, creating dataset...")
        train_data, val_data, _ = build_dataset(num_samples=1000)
    
    # Update vocab size based on data
    if train_data and "input_ids" in train_data:
        max_token_id = max(max(seq) for seq in train_data["input_ids"])
        model_config.vocab_size = max_token_id + 100  # Add buffer
    
    # Create model
    model = SymptomDiagnosisGPT(model_config)
    model = model.to(device)
    
    # Wrap model for distributed training
    model = train.torch.prepare_model(model)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, model_config,
        rank=train.get_context().get_world_rank(),
        world_size=train.get_context().get_world_size()
    )
    
    # Prepare data loaders for Ray
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay
    )
    
    # Cosine annealing scheduler
    total_steps = len(train_loader) * model_config.max_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    for epoch in range(model_config.max_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, model_config, epoch, 
            rank=train.get_context().get_world_rank()
        )
        
        # Validate
        val_metrics = compute_metrics(model, val_loader, device)
        
        # Report metrics to Ray
        train.report({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"]
        })
        
        # Save checkpoint
        if epoch % distributed_config.checkpoint_freq == 0:
            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"]
            }
            train.report(metrics={}, checkpoint=train.Checkpoint.from_dict(checkpoint_dict))


def setup_ddp(rank, world_size, backend="nccl"):
    """Setup PyTorch DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use gloo for CPU, nccl for GPU
    if not torch.cuda.is_available():
        backend = "gloo"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup PyTorch DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_train_worker(rank, world_size, config_dict):
    """DDP training worker function."""
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        
        # Get configurations
        model_config = config_dict["model_config"]
        distributed_config = config_dict["distributed_config"]
        
        # Setup device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Worker {rank}: Using device {device}")
        
        # Load data
        train_data, val_data, _ = load_dataset()
        if train_data is None:
            if rank == 0:
                logger.info("No processed data found, creating dataset...")
                train_data, val_data, _ = build_dataset(num_samples=1000)
            else:
                # Wait for rank 0 to create data
                time.sleep(10)
                train_data, val_data, _ = load_dataset()
        
        # Update vocab size based on data
        if train_data and "input_ids" in train_data:
            max_token_id = max(max(seq) for seq in train_data["input_ids"])
            model_config.vocab_size = max_token_id + 100
        
        # Create model
        model = SymptomDiagnosisGPT(model_config)
        model = model.to(device)
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model)
        
        # Create data loaders with distributed sampling
        train_loader, val_loader = create_data_loaders(
            train_data, val_data, model_config, rank, world_size
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay
        )
        
        total_steps = len(train_loader) * model_config.max_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(model_config.max_epochs):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                device, model_config, epoch, rank
            )
            
            # Validate (only on rank 0)
            if rank == 0:
                val_metrics = compute_metrics(model, val_loader, device)
                
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                           f"val_loss={val_metrics['loss']:.4f}, "
                           f"val_accuracy={val_metrics['accuracy']:.4f}")
                
                # Save checkpoint
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    
                    # Save model (unwrap DDP)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_checkpoint(
                        model_config.model_save_path,
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                        loss=val_metrics['loss']
                    )
        
        logger.info(f"Worker {rank}: Training completed!")
        
    except Exception as e:
        logger.error(f"Worker {rank} failed: {e}")
        raise
    finally:
        cleanup_ddp()


def train_with_pytorch_ddp(model_config, distributed_config):
    """Train using PyTorch DDP."""
    if not TORCH_DDP_AVAILABLE:
        logger.warning("PyTorch DDP not available, falling back to single-node")
        return None
    
    world_size = model_config.num_workers
    
    logger.info(f"🚀 Starting PyTorch DDP training with {world_size} workers...")
    
    # Prepare config for workers
    config_dict = {
        "model_config": model_config,
        "distributed_config": distributed_config
    }
    
    try:
        # Spawn processes for DDP
        mp.spawn(
            ddp_train_worker,
            args=(world_size, config_dict),
            nprocs=world_size,
            join=True
        )
        
        logger.info("✅ PyTorch DDP training completed!")
        return {"status": "success", "method": "pytorch_ddp"}
        
    except Exception as e:
        logger.error(f"PyTorch DDP training failed: {e}")
        return None


class DistributedTrainer:
    """Main distributed training coordinator."""
    
    def __init__(self, model_config=None, distributed_config=None):
        self.model_config = model_config or get_model_config()
        self.distributed_config = distributed_config or get_distributed_config()
        
    def train(self):
        """Main training entry point with automatic fallback."""
        logger.info("🎯 Starting distributed training...")
        
        # Try PyTorch DDP first (more reliable and doesn't require Ray)
        if TORCH_DDP_AVAILABLE and self.model_config.num_workers > 1:
            logger.info("Attempting PyTorch DDP training...")
            result = train_with_pytorch_ddp(self.model_config, self.distributed_config)
            if result:
                return result
        
        # Try Ray if available and requested
        if RAY_AVAILABLE and self.model_config.use_ray:
            logger.info("Attempting Ray distributed training...")
            result = self.train_with_ray()
            if result:
                return result
        
        # Fallback to single-node
        logger.info("Falling back to single-node training...")
        return self.train_single_node()
        
    def setup_ray(self):
        """Initialize Ray cluster."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, cannot use distributed training")
            return False
        
        try:
            if not ray.is_initialized():
                if self.distributed_config.ray_address:
                    # Connect to existing cluster
                    ray.init(address=self.distributed_config.ray_address)
                    logger.info(f"Connected to Ray cluster: {self.distributed_config.ray_address}")
                else:
                    # Start local cluster
                    ray.init(
                        num_cpus=self.distributed_config.num_cpus_per_worker * self.model_config.num_workers,
                        num_gpus=self.distributed_config.num_gpus_per_worker * self.model_config.num_workers
                    )
                    logger.info("Started local Ray cluster")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False
    
    def train_with_ray(self):
        """Train using Ray distributed training."""
        if not self.setup_ray():
            return self.train_single_node()
        
        try:
            # Configure Ray trainer
            scaling_config = ScalingConfig(
                num_workers=self.model_config.num_workers,
                use_gpu=self.distributed_config.num_gpus_per_worker > 0,
                resources_per_worker={
                    "CPU": self.distributed_config.num_cpus_per_worker,
                    "GPU": self.distributed_config.num_gpus_per_worker
                }
            )
            
            run_config = RunConfig(
                name="symptom-diagnosis-gpt",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute="val_loss",
                    checkpoint_score_order="min"
                )
            )
            
            # Create trainer
            trainer = TorchTrainer(
                train_loop_per_worker=ray_train_func,
                train_loop_config={
                    "model_config": self.model_config,
                    "distributed_config": self.distributed_config
                },
                scaling_config=scaling_config,
                run_config=run_config
            )
            
            # Start training
            logger.info("🚀 Starting Ray distributed training...")
            result = trainer.fit()
            
            # Save final model
            self.save_final_model(result)
            
            logger.info("✅ Ray distributed training completed!")
            return result
            
        except Exception as e:
            logger.error(f"Ray training failed: {e}")
            logger.info("Falling back to single-node training...")
            return self.train_single_node()
        
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def train_single_node(self):
        """Fallback single-node training."""
        logger.info("🔄 Starting single-node training...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load or create data
        train_data, val_data, _ = load_dataset()
        if train_data is None:
            logger.info("No processed data found, creating dataset...")
            train_data, val_data, _ = build_dataset(num_samples=1000)
        
        # Update vocab size
        if train_data and "input_ids" in train_data:
            max_token_id = max(max(seq) for seq in train_data["input_ids"])
            self.model_config.vocab_size = max_token_id + 100
        
        # Create model
        model = SymptomDiagnosisGPT(self.model_config)
        model = model.to(device)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_data, val_data, self.model_config
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        
        total_steps = len(train_loader) * self.model_config.max_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.model_config.max_epochs):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                device, self.model_config, epoch
            )
            
            # Validate
            val_metrics = compute_metrics(model, val_loader, device)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_metrics['loss']:.4f}, "
                       f"val_accuracy={val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                model.save_checkpoint(
                    self.model_config.model_save_path,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    loss=val_metrics['loss']
                )
        
        logger.info("✅ Single-node training completed!")
        return {"best_val_loss": best_val_loss}
    
    def save_final_model(self, result):
        """Save the final trained model."""
        try:
            # Get best checkpoint
            checkpoint = result.best_checkpoints[0][1]
            
            # Load checkpoint data
            checkpoint_data = checkpoint.to_dict()
            
            # Save model
            os.makedirs(os.path.dirname(self.model_config.model_save_path), exist_ok=True)
            torch.save(checkpoint_data, self.model_config.model_save_path)
            
            logger.info(f"✅ Final model saved to {self.model_config.model_save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")


def main():
    """Main function for distributed training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed training for Symptom-Diagnosis-GPT")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use-ray", action="store_true", default=False, help="Force use Ray (if available)")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--data-samples", type=int, default=1000, help="Number of data samples to generate")
    parser.add_argument("--single-node", action="store_true", help="Force single-node training")
    
    args = parser.parse_args()
    
    # Print available training methods
    print("🎯 Available Training Methods:")
    print(f"   - PyTorch DDP: {'✅' if TORCH_DDP_AVAILABLE else '❌'}")
    print(f"   - Ray: {'✅' if RAY_AVAILABLE else '❌'}")
    print(f"   - Single-node: ✅")
    print()
    
    args = parser.parse_args()
    
    # Update configuration
    model_config = get_model_config()
    distributed_config = get_distributed_config()
    
    model_config.num_workers = args.num_workers
    model_config.max_epochs = args.num_epochs
    model_config.batch_size = args.batch_size
    model_config.learning_rate = args.learning_rate
    model_config.use_ray = args.use_ray
    distributed_config.ray_address = args.ray_address
    
    # Force single-node if requested
    if args.single_node:
        model_config.num_workers = 1
        model_config.use_ray = False
    
    # Ensure data exists
    train_data, val_data, _ = load_dataset()
    if train_data is None:
        logger.info(f"Creating dataset with {args.data_samples} samples...")
        build_dataset(num_samples=args.data_samples)
    
    # Start training
    trainer = DistributedTrainer(model_config, distributed_config)
    result = trainer.train()
    
    logger.info(f"Training completed with result: {result}")


if __name__ == "__main__":
    main()