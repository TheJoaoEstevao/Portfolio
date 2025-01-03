import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    attention_type: str = "efficient"  # [efficient, flash, sparse]
    max_sequence_length: int = 2048
    use_alibi: bool = False  # Position embedding alternative
    use_rope: bool = True   # Rotary position embedding
    attention_window: int = 512  # For sparse attention

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding implementation"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
            self.cos_cached = emb.cos()[:, :seq_len, ...]
            self.sin_cached = emb.sin()[:, :seq_len, ...]
        return self.cos_cached, self.sin_cached

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class EfficientAttention(nn.Module):
    """Memory-efficient attention implementation"""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Rotary embeddings if enabled
        self.rotary = RotaryEmbedding(self.head_dim) if config.use_rope else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape and transpose for attention
        query_states = query_states.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.rotary is not None:
            seq_len = key_states.shape[2]
            cos, sin = self.rotary(value_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Efficient attention computation
        attn_output = self._efficient_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, -1, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

    def _efficient_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Memory-efficient attention computation"""
        
        # Scaled dot product attention
        scale_factor = math.sqrt(self.head_dim)
        
        # Chunk-based attention computation
        chunk_size = min(self.config.attention_window, query_states.shape[2])
        num_chunks = (query_states.shape[2] + chunk_size - 1) // chunk_size
        
        attention_outputs = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, query_states.shape[2])
            
            chunk_query = query_states[:, :, start_idx:end_idx]
            chunk_attn_weights = torch.matmul(chunk_query, key_states.transpose(-1, -2)) / scale_factor
            
            if attention_mask is not None:
                chunk_attn_weights = chunk_attn_weights + attention_mask[:, :, start_idx:end_idx, :]
            
            chunk_attn_weights = nn.functional.softmax(chunk_attn_weights, dim=-1)
            chunk_attn_weights = self.dropout(chunk_attn_weights)
            
            chunk_attn_output = torch.matmul(chunk_attn_weights, value_states)
            attention_outputs.append(chunk_attn_output)
        
        return torch.cat(attention_outputs, dim=2)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to input tensors."""
    # Reshape for broadcasting
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Apply rotation using complex number multiplication
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class FlashAttention(EfficientAttention):
    """Flash Attention implementation for faster training"""
    def _efficient_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash attention computation"""
        # Note: This is a placeholder for actual flash attention
        # Real implementation would use CUDA kernels for more efficiency
        scale_factor = math.sqrt(self.head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / scale_factor
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, value_states)

class SparseAttention(EfficientAttention):
    """Sparse Attention for handling long sequences"""
    def _efficient_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse attention computation using local windows"""
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        window_size = self.config.attention_window
        
        # Reshape into windows
        padded_length = ((seq_length + window_size - 1) // window_size) * window_size
        padding_length = padded_length - seq_length
        
        if padding_length > 0:
            query_states = torch.nn.functional.pad(query_states, (0, 0, 0, padding_length))
            key_states = torch.nn.functional.pad(key_states, (0, 0, 0, padding_length))
            value_states = torch.nn.functional.pad(value_states, (0, 0, 0, padding_length))
            
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, padding_length, 0, padding_length), value=float("-inf")
                )
        
        # Compute attention for each window
        query_windows = query_states.view(batch_size, num_heads, -1, window_size, head_dim)
        key_windows = key_states.view(batch_size, num_heads, -1, window_size, head_dim)
        value_windows = value_states.view(batch_size, num_heads, -1, window_size, head_dim)
        
        scale_factor = math.sqrt(head_dim)
        attn_weights = torch.matmul(query_windows, key_windows.transpose(-1, -2)) / scale_factor
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, num_heads, -1, window_size, window_size)
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_windows)
        
        # Reshape back
        attn_output = attn_output.view(batch_size, num_heads, padded_length, head_dim)
        if padding_length > 0:
            attn_output = attn_output[:, :, :seq_length, :]
            
        return attn_output 