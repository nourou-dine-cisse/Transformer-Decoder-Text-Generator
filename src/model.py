import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope_embedding import RoPE


def causal_mask(seq_len: int, past_len: int = 0, device=None) -> torch.Tensor:
    """Creates a causal attention mask compatible with past key/value cache."""
    query_positions = past_len + torch.arange(seq_len, device=device)
    key_positions = torch.arange(past_len + seq_len, device=device)

    mask = torch.zeros(seq_len, past_len + seq_len, device=device)
    mask = mask.masked_fill(
        key_positions.unsqueeze(0) > query_positions.unsqueeze(1),
        float("-inf"),
    )
    return mask


def multi_head_attention(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    mask: torch.FloatTensor = None,
):
    """
    Given Q, K, V and an attention mask, compute scaled dot-product multi-head attention.
    
    Shapes:
        q: (B, H, L_q, D_h)
        k: (B, H, L_k, D_h)
        v: (B, H, L_k, D_h)
        mask: (L_q, L_k) or (B, L_q, L_k) with -inf on masked positions
    """
    scale = math.sqrt(q.size(-1))
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        mask = mask.to(device=attention_scores.device, dtype=attention_scores.dtype)
        attention_scores = attention_scores + mask

    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_weights, attention_output


class RoPEAttentionModule(nn.Module):
    """
    Multi-Head Attention module with Rotary Position Embeddings (RoPE)
    and optional Key/Value caching for autoregressive decoding.
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.Wq = nn.Linear(input_dim, output_dim * num_heads)
        self.Wk = nn.Linear(input_dim, output_dim * num_heads)
        self.Wv = nn.Linear(input_dim, output_dim * num_heads)
        self.Wo = nn.Linear(output_dim * num_heads, input_dim)
        self.rope_func = RoPE(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.BoolTensor = None,
        y: torch.Tensor = None,
        decoder_mask: bool = True,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        start_cache: int = 0,
    ):
        B, L = x.size(0), x.size(1)
        H, dh = self.num_heads, self.output_dim

        if decoder_mask:
            q = self.Wq(x).reshape(B, L, H, dh).transpose(1, 2)
            k_new = self.Wk(x).reshape(B, L, H, dh).transpose(1, 2)
            v_new = self.Wv(x).reshape(B, L, H, dh).transpose(1, 2)

            q = self.rope_func(q, start_pos_id=start_cache)
            k_new = self.rope_func(k_new, start_pos_id=start_cache)

            past_len = 0 if k_cache is None else k_cache.size(2)

            if k_cache is not None:
                k = torch.cat([k_cache, k_new], dim=2)
                v = torch.cat([v_cache, v_new], dim=2)
            else:
                k, v = k_new, v_new

            mask = causal_mask(L, past_len=past_len, device=x.device)

            weights, attn_out = multi_head_attention(q, k, v, mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, L, H * dh)
            output = self.Wo(attn_out)
            return weights, output, k, v

        # Cross-attention (if y is provided)
        k = self.Wk(x).reshape(B, L, H, dh).transpose(1, 2)
        v = self.Wv(x).reshape(B, L, H, dh).transpose(1, 2)

        Ly = y.size(1)
        q = self.Wq(y).reshape(B, Ly, H, dh).transpose(1, 2)
        q = self.rope_func(q)
        k = self.rope_func(k)

        weights, attn_out = multi_head_attention(q, k, v, attention_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, Ly, H * dh)
        output = self.Wo(attn_out)
        return weights, output


class TransformerFeedForward(nn.Module):
    """FeedForward network following LLaMA-style SwiGLU gating."""
    def __init__(self, embed_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_size, intermediate_size)
        self.up_proj = nn.Linear(intermediate_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, embed_size)
        self.gate_func = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_proj(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return self.gate_func(x)


class TransformerDecoderBlock(nn.Module):
    """Decoder block with RMSNorm Pre-Norm, RoPE Multi-Head Attention, and SwiGLU FFN."""
    def __init__(self, embed_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.attention_module = RoPEAttentionModule(embed_size, embed_size, num_heads=num_heads)
        self.feed_forward = TransformerFeedForward(embed_size, intermediate_size)
        self.attention_layer_norm = nn.RMSNorm(embed_size)
        self.feed_forward_layer_norm = nn.RMSNorm(embed_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        k_cache=None,
        v_cache=None,
        start_cache: int = 0,
        layer: int = 0,
    ):
        x1 = self.attention_layer_norm(x)
        layer_k = None if k_cache is None else k_cache[layer]
        layer_v = None if v_cache is None else v_cache[layer]

        _, x2, new_k, new_v = self.attention_module(
            x1,
            attention_mask=attention_mask,
            decoder_mask=True,
            k_cache=layer_k,
            v_cache=layer_v,
            start_cache=start_cache,
        )

        x3 = x + x2
        x4 = self.feed_forward_layer_norm(x3)
        output = self.feed_forward(x4) + x3
        return output, new_k, new_v


class TransformerDecoder(nn.Module):
    """
    Autoregressive Decoder-Only Transformer architecture (LLaMA-2 inspired).
    Features:
      - Token embeddings
      - Pre-norm residual blocks with RoPE Attention & SwiGLU FeedForward
      - Final LayerNorm and Linear LM head
      - Built-in KV-caching support
    """
    def __init__(
        self,
        vocabulary_size: int,
        embed_size: int = 128,
        intermediate_size: int = 256,
        num_heads: int = 4,
        hidden_layers: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nh = num_heads
        self.nhl = hidden_layers
        self.es = embed_size

        self.wte = nn.Embedding(vocabulary_size, embed_size)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_size, intermediate_size, num_heads)
            for _ in range(hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size, eps=1e-3)
        self.lm_head = nn.Linear(embed_size, vocabulary_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        k_cache=None,
        v_cache=None,
        start_cache: int = 0,
    ):
        input_embed = self.wte(input_ids)
        intermediate = input_embed

        if k_cache is None:
            k_cache = [None] * self.nhl
        if v_cache is None:
            v_cache = [None] * self.nhl

        new_k_cache, new_v_cache = [], []

        for i, block in enumerate(self.blocks):
            intermediate, layer_k, layer_v = block(
                intermediate,
                attention_mask=attention_mask,
                k_cache=k_cache,
                v_cache=v_cache,
                start_cache=start_cache,
                layer=i,
            )
            new_k_cache.append(layer_k)
            new_v_cache.append(layer_v)

        output_embed = self.ln_f(intermediate)
        output_lm = self.lm_head(output_embed)
        return output_embed, output_lm, new_k_cache, new_v_cache
