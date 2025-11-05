"""
GPT-like transformer model for symptom-diagnosis prediction.
Lightweight architecture optimized for distributed training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import get_model_config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, n_embed: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, n_embed)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * 
                           (-math.log(10000.0) / n_embed))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_embed % n_heads == 0
        
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        
        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.out = nn.Linear(n_embed, n_embed)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention."""
        batch_size, seq_len, n_embed = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
        out = self.out(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, n_embed: int, n_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, n_ff)
        self.linear2 = nn.Linear(n_ff, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(n_embed, n_heads, dropout)
        self.feed_forward = FeedForward(n_embed, n_embed * 4, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x


class SymptomDiagnosisGPT(nn.Module):
    """
    GPT-like transformer model for symptom-diagnosis prediction.
    Lightweight architecture optimized for distributed training.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or get_model_config()
        
        # Model components
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.positional_encoding = PositionalEncoding(
            self.config.n_embed, 
            self.config.max_length, 
            self.config.dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                self.config.n_embed,
                self.config.n_heads,
                self.config.dropout
            ) for _ in range(self.config.n_layers)
        ])
        
        # Output layers
        self.ln_final = nn.LayerNorm(self.config.n_embed)
        self.head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔧 Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask to ignore padding tokens."""
        # Assume padding token is 0
        return (input_ids != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
        
        Returns:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            loss: Computed loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, n_embed]
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            # Ignore padding tokens (label = 0) in loss computation
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)
        
        return logits, loss
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text continuation given input tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for next token
                logits, _ = self.forward(input_ids)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if sequence gets too long
                if input_ids.size(1) >= self.config.max_length:
                    break
        
        return input_ids
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def save_checkpoint(self, filepath: str, optimizer_state=None, epoch=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, map_location=None):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Create model with saved config
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded from {filepath}")
        return model, checkpoint


# Legacy compatibility
class GPTConfig:
    """Legacy config class for backward compatibility."""
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPTModel(nn.Module):
    """Legacy model class for backward compatibility."""
    def __init__(self, vocab_size):
        super().__init__()
        config = get_model_config()
        
        self.embed = nn.Embedding(vocab_size, config.n_embed)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embed,
                nhead=config.n_heads,
                dim_feedforward=config.n_embed*4,
                dropout=config.dropout
            ),
            num_layers=config.n_layers
        )
        self.ln = nn.LayerNorm(config.n_embed)
        self.fc = nn.Linear(config.n_embed, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.fc(x)
        return logits
