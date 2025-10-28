#!/usr/bin/env python3
"""
MLX TrOCR Model Implementation
Vision-Encoder-Decoder architecture using MLX for Apple Silicon

Architecture:
- Encoder: Vision Transformer (ViT) - processes images
- Decoder: GPT-2 - generates text

Author: Diego Alarcon
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple
import math


# ============================================
# Vision Transformer (ViT) Encoder Components
# ============================================

class PatchEmbeddings(nn.Module):
    """
    Convert image to patch embeddings
    Image (H, W, C) -> Patches (num_patches, hidden_size)
    """
    def __init__(self, image_size=384, patch_size=16, num_channels=3, hidden_size=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolution to create patches
        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def __call__(self, x):
        # x: (batch, height, width, channels)
        # Conv2d expects (batch, channels, height, width)
        x = mx.transpose(x, (0, 3, 1, 2))  # BHWC -> BCHW
        x = self.projection(x)  # (batch, hidden_size, h_patches, w_patches)

        # Flatten patches
        batch_size, hidden_size, h, w = x.shape
        x = mx.transpose(x, (0, 2, 3, 1))  # BCHW -> BHWC
        x = mx.reshape(x, (batch_size, h * w, hidden_size))

        return x


class ViTEmbeddings(nn.Module):
    """
    Complete embeddings: patches + position embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_channels=3,
            hidden_size=config["hidden_size"]
        )

        num_patches = self.patch_embeddings.num_patches

        # CLS token
        self.cls_token = mx.zeros((1, 1, config["hidden_size"]))

        # Position embeddings (1 for CLS + num_patches)
        self.position_embeddings = mx.zeros((1, num_patches + 1, config["hidden_size"]))

    def __call__(self, x):
        batch_size = x.shape[0]

        # Get patch embeddings
        embeddings = self.patch_embeddings(x)

        # Expand CLS token for batch
        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.cls_token.shape[-1]))

        # Concatenate CLS token
        embeddings = mx.concatenate([cls_tokens, embeddings], axis=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embeddings

        return embeddings


class ViTSelfAttention(nn.Module):
    """Multi-head self-attention for ViT"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # Project to Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Attention scores
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)

        # Reshape back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, hidden_size))

        return attn_output


class ViTSelfOutput(nn.Module):
    """Output projection after self-attention"""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states):
        return self.dense(hidden_states)


class ViTAttention(nn.Module):
    """Complete attention block with residual"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = ViTSelfAttention(hidden_size, num_heads)
        self.output = ViTSelfOutput(hidden_size)

    def __call__(self, x):
        attn_output = self.attention(x)
        attn_output = self.output(attn_output)
        return attn_output


class ViTMLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class ViTLayer(nn.Module):
    """Single transformer layer"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = ViTAttention(hidden_size, num_heads)
        self.mlp = ViTMLP(hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.layernorm_after = nn.LayerNorm(hidden_size)

    def __call__(self, x):
        # Attention with residual
        residual = x
        x = self.layernorm_before(x)
        x = self.attention(x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.layernorm_after(x)
        x = self.mlp(x)
        x = x + residual

        return x


class ViTEncoder(nn.Module):
    """Complete ViT encoder"""
    def __init__(self, config):
        super().__init__()
        self.layers = [
            ViTLayer(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"]
            )
            for _ in range(config["num_layers"])
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """Complete Vision Transformer"""
    def __init__(self, config):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config["hidden_size"])

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.layernorm(x)
        return x


# ============================================
# GPT-2 Decoder Components
# ============================================

class GPT2Attention(nn.Module):
    """GPT-2 causal self-attention"""
    def __init__(self, hidden_size, num_heads, max_length=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)

        # Causal mask
        self.max_length = max_length

    def __call__(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)
        qkv = mx.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = mx.split(qkv, 3, axis=2)

        q = mx.squeeze(q, axis=2)
        k = mx.squeeze(k, axis=2)
        v = mx.squeeze(v, axis=2)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Attention scores
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply causal mask
        causal_mask = mx.tril(mx.ones((seq_len, seq_len)))
        scores = mx.where(causal_mask, scores, -1e9)

        attn_weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)

        # Reshape back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, hidden_size))

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output


class GPT2MLP(nn.Module):
    """GPT-2 feed-forward network"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    """Single GPT-2 transformer block"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = GPT2Attention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = GPT2MLP(hidden_size, intermediate_size)

    def __call__(self, x):
        # Attention with residual
        residual = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class GPT2Decoder(nn.Module):
    """GPT-2 decoder for text generation"""
    def __init__(self, config):
        super().__init__()

        # Token embeddings
        self.wte = nn.Embedding(config["vocab_size"], config["hidden_size"])

        # Position embeddings
        self.wpe = nn.Embedding(config["max_length"], config["hidden_size"])

        # Transformer blocks
        self.h = [
            GPT2Block(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                intermediate_size=config["intermediate_size"]
            )
            for _ in range(config["num_layers"])
        ]

        self.ln_f = nn.LayerNorm(config["hidden_size"])

    def __call__(self, input_ids, encoder_hidden_states=None):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        inputs_embeds = self.wte(input_ids)

        # Position embeddings
        position_ids = mx.arange(seq_len)[None, :]
        position_embeds = self.wpe(position_ids)

        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds

        # Apply transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


# ============================================
# Complete TrOCR Model
# ============================================

class MLXTrOCR(nn.Module):
    """
    Complete TrOCR model in MLX
    Vision Encoder + Text Decoder
    """
    def __init__(self, encoder_config, decoder_config):
        super().__init__()

        self.encoder = VisionTransformer(encoder_config)
        self.decoder = GPT2Decoder(decoder_config)

        # LM head for vocabulary prediction
        self.lm_head = nn.Linear(decoder_config["hidden_size"], decoder_config["vocab_size"], bias=False)

    def encode(self, pixel_values):
        """Encode image to hidden states"""
        return self.encoder(pixel_values)

    def decode(self, input_ids, encoder_hidden_states):
        """Decode text from encoder hidden states"""
        hidden_states = self.decoder(input_ids, encoder_hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def __call__(self, pixel_values, input_ids):
        """Full forward pass"""
        encoder_hidden_states = self.encode(pixel_values)
        logits = self.decode(input_ids, encoder_hidden_states)
        return logits

    def generate(self, pixel_values, max_length=64, temperature=1.0):
        """
        Generate text from image
        Simple greedy decoding (can be enhanced with beam search)
        """
        # Encode image
        encoder_hidden_states = self.encode(pixel_values)

        batch_size = pixel_values.shape[0]

        # Start with BOS token (typically 0)
        input_ids = mx.zeros((batch_size, 1), dtype=mx.int32)

        for _ in range(max_length):
            # Get logits for next token
            logits = self.decode(input_ids, encoder_hidden_states)

            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Greedy sampling
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)

            # Append to sequence
            input_ids = mx.concatenate([input_ids, next_token], axis=1)

            # Check for EOS token (typically 2)
            if mx.all(next_token == 2):
                break

        return input_ids


def create_trocr_base():
    """
    Create TrOCR base model with standard configuration
    Matches microsoft/trocr-base-handwritten
    """
    encoder_config = {
        "image_size": 384,
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
    }

    decoder_config = {
        "vocab_size": 50265,  # RoBERTa tokenizer
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "max_length": 512,
    }

    return MLXTrOCR(encoder_config, decoder_config)


if __name__ == "__main__":
    print("Creating MLX TrOCR model...")
    model = create_trocr_base()

    print("\nModel structure:")
    print(model)

    print("\n✓ MLX TrOCR model created successfully!")
