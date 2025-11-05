"""
Legacy training script for Symptom-Diagnosis-GPT.
This is a simplified single-node training script.
For distributed training, use train_distributed.py instead.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from .config import get_model_config
from .model import SymptomDiagnosisGPT
from .prepare_data import load_dataset, build_dataset, SymptomDataset
from .train_distributed import compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(vocab_size=None, use_real_data=True):
    """
    Train the model using single-node training.
    
    Args:
        vocab_size: Vocabulary size (auto-detected if None)
        use_real_data: Whether to use real dataset or dummy data
    """
    config = get_model_config()
    
    logger.info("🔄 Starting single-node training...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if use_real_data:
        # Load or create real dataset
        train_data, val_data, _ = load_dataset()
        
        if train_data is None:
            logger.info("No processed data found, creating dataset...")
            train_data, val_data, _ = build_dataset(num_samples=1000)
        
        # Update vocab size based on data
        if train_data and "input_ids" in train_data:
            max_token_id = max(max(seq) for seq in train_data["input_ids"])
            config.vocab_size = max_token_id + 100  # Add buffer
        
        # Create datasets and loaders
        train_dataset = SymptomDataset(train_data)
        val_dataset = SymptomDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
    else:
        # Use dummy data for quick testing
        vocab_size = vocab_size or 1000
        config.vocab_size = vocab_size
        train_loader = None
        val_loader = None
        logger.info("Using dummy data for testing")
    
    # Create model
    model = SymptomDiagnosisGPT(config).to(device)
    logger.info(f"Model created with {model.get_num_params():,} parameters")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    if use_real_data:
        # Train with real data
        total_steps = len(train_loader) * config.max_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        for epoch in range(config.max_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits, loss = model(input_ids, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % config.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}")
            
            # Validation
            val_metrics = compute_metrics(model, val_loader, device)
            avg_train_loss = epoch_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                       f"val_loss={val_metrics['loss']:.4f}, "
                       f"val_accuracy={val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
                model.save_checkpoint(
                    config.model_save_path,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    loss=val_metrics['loss']
                )
                logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")
    
    else:
        # Train with dummy data (for testing)
        num_steps = 100
        
        for step in range(num_steps):
            # Create dummy batch
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_length)).to(device)
            y = torch.randint(0, config.vocab_size, (config.batch_size, config.max_length)).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"Step {step} | Loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        model.save_checkpoint(
            config.model_save_path,
            optimizer_state=optimizer.state_dict(),
            epoch=0,
            loss=loss.item()
        )
    
    logger.info(f"✅ Training completed! Model saved to {config.model_save_path}")
    return model


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Symptom-Diagnosis-GPT (single-node)")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--dummy-data", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Update config
    config = get_model_config()
    config.max_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    
    # Train model
    train_model(
        vocab_size=args.vocab_size,
        use_real_data=not args.dummy_data
    )


if __name__ == "__main__":
    main()
