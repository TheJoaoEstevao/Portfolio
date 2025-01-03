import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class EfficientConfig:
    """Configuration for efficient transformer"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # Grouped-query attention
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    sliding_window: int = 256
    num_layers: int = 12

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - More efficient than learned positional embeddings"""
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotary embedding helper"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    """Grouped-query attention for more efficient computation"""
    def __init__(self, config: EfficientConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.sliding_window = config.sliding_window

        # Linear layers
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(value_states, seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads to match number of query heads
        key_states = torch.repeat_interleave(key_states, self.num_attention_heads // self.num_key_value_heads, dim=2)
        value_states = torch.repeat_interleave(value_states, self.num_attention_heads // self.num_key_value_heads, dim=2)

        # Efficient attention with sliding window
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        
        # Process in sliding windows for memory efficiency
        attention_output = []
        for i in range(0, seq_length, self.sliding_window):
            window_end = min(i + self.sliding_window, seq_length)
            q_window = query_states[:, i:window_end]
            
            # Compute attention scores for the window
            attn_weights = torch.matmul(q_window, key_states.transpose(-1, -2)) * scale_factor
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, i:window_end, :]
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value_states)
            attention_output.append(attn_output)
            
        # Concatenate window outputs
        attention_output = torch.cat(attention_output, dim=1)
        
        # Reshape and project output
        attention_output = attention_output.reshape(batch_size, seq_length, self.hidden_size)
        attention_output = self.o_proj(attention_output)
        
        return attention_output

class EfficientMLP(nn.Module):
    """Efficient MLP with SwiGLU activation"""
    def __init__(self, config: EfficientConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block with modern optimizations"""
    def __init__(self, config: EfficientConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.mlp = EfficientMLP(config)
        
        # RMSNorm is more efficient than LayerNorm
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        attn_output = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            position_ids,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(mlp_output)
        
        return hidden_states

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - More efficient than standard LayerNorm"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class EfficientTransformer(nn.Module):
    """Modern efficient transformer with latest optimizations"""
    def __init__(self, config: EfficientConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings only - position handled by rotary
        self.embed_tokens = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

def create_efficient_transformer(config: EfficientConfig) -> nn.Module:
    """Create an efficient transformer model"""
    return EfficientTransformer(config) 