#!/usr/bin/env python3
"""
Convert HuggingFace TrOCR model to MLX format

This script:
1. Loads a PyTorch TrOCR model from HuggingFace
2. Converts weights to MLX format
3. Saves the MLX model for inference

Author: Diego Alarcon
Date: October 2025
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_models import create_trocr_base, MLXTrOCR
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


def convert_pytorch_to_numpy(state_dict):
    """Convert PyTorch state dict to numpy arrays"""
    numpy_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.cpu().detach().numpy()
        else:
            numpy_dict[key] = value
    return numpy_dict


def map_weights(pytorch_state, mlx_model):
    """
    Map PyTorch weights to MLX model
    This is the critical part - matching layer names and shapes
    """
    print("\n" + "=" * 70)
    print("WEIGHT MAPPING")
    print("=" * 70)

    # Get PyTorch state dict as numpy
    pt_weights = convert_pytorch_to_numpy(pytorch_state)

    # Get MLX model parameters
    mlx_params = mlx_model.parameters()

    # Weight mapping dictionary (PyTorch -> MLX)
    weight_map = {}

    # ============================================
    # ENCODER MAPPINGS (ViT)
    # ============================================

    # Patch embeddings
    weight_map["encoder.embeddings.patch_embeddings.projection.weight"] = \
        "encoder.encoder.embeddings.patch_embeddings.projection.weight"
    weight_map["encoder.embeddings.patch_embeddings.projection.bias"] = \
        "encoder.encoder.embeddings.patch_embeddings.projection.bias"

    # Position embeddings
    weight_map["encoder.embeddings.position_embeddings"] = \
        "encoder.encoder.embeddings.position_embeddings"

    # CLS token
    weight_map["encoder.embeddings.cls_token"] = \
        "encoder.encoder.embeddings.cls_token"

    # Encoder layers
    for i in range(12):  # 12 layers for base model
        prefix_pt = f"encoder.encoder.layer.{i}"
        prefix_mlx = f"encoder.encoder.layers.{i}"

        # Attention
        weight_map[f"{prefix_pt}.attention.attention.query.weight"] = \
            f"{prefix_mlx}.attention.attention.query.weight"
        weight_map[f"{prefix_pt}.attention.attention.query.bias"] = \
            f"{prefix_mlx}.attention.attention.query.bias"
        weight_map[f"{prefix_pt}.attention.attention.key.weight"] = \
            f"{prefix_mlx}.attention.attention.key.weight"
        weight_map[f"{prefix_pt}.attention.attention.key.bias"] = \
            f"{prefix_mlx}.attention.attention.key.bias"
        weight_map[f"{prefix_pt}.attention.attention.value.weight"] = \
            f"{prefix_mlx}.attention.attention.value.weight"
        weight_map[f"{prefix_pt}.attention.attention.value.bias"] = \
            f"{prefix_mlx}.attention.attention.value.bias"

        # Attention output
        weight_map[f"{prefix_pt}.attention.output.dense.weight"] = \
            f"{prefix_mlx}.attention.output.dense.weight"
        weight_map[f"{prefix_pt}.attention.output.dense.bias"] = \
            f"{prefix_mlx}.attention.output.dense.bias"

        # Layer norms
        weight_map[f"{prefix_pt}.layernorm_before.weight"] = \
            f"{prefix_mlx}.layernorm_before.weight"
        weight_map[f"{prefix_pt}.layernorm_before.bias"] = \
            f"{prefix_mlx}.layernorm_before.bias"
        weight_map[f"{prefix_pt}.layernorm_after.weight"] = \
            f"{prefix_mlx}.layernorm_after.weight"
        weight_map[f"{prefix_pt}.layernorm_after.bias"] = \
            f"{prefix_mlx}.layernorm_after.bias"

        # MLP
        weight_map[f"{prefix_pt}.intermediate.dense.weight"] = \
            f"{prefix_mlx}.mlp.fc1.weight"
        weight_map[f"{prefix_pt}.intermediate.dense.bias"] = \
            f"{prefix_mlx}.mlp.fc1.bias"
        weight_map[f"{prefix_pt}.output.dense.weight"] = \
            f"{prefix_mlx}.mlp.fc2.weight"
        weight_map[f"{prefix_pt}.output.dense.bias"] = \
            f"{prefix_mlx}.mlp.fc2.bias"

    # Final layer norm
    weight_map["encoder.encoder.layernorm.weight"] = \
        "encoder.encoder.layernorm.weight"
    weight_map["encoder.encoder.layernorm.bias"] = \
        "encoder.encoder.layernorm.bias"

    # ============================================
    # DECODER MAPPINGS (GPT-2)
    # ============================================

    # Embeddings
    weight_map["decoder.model.decoder.embed_tokens.weight"] = \
        "decoder.decoder.wte.weight"
    weight_map["decoder.model.decoder.embed_positions.weight"] = \
        "decoder.decoder.wpe.weight"

    # Decoder layers
    for i in range(12):  # 12 layers for base model
        prefix_pt = f"decoder.model.decoder.layers.{i}"
        prefix_mlx = f"decoder.decoder.h.{i}"

        # Layer norms
        weight_map[f"{prefix_pt}.self_attn_layer_norm.weight"] = \
            f"{prefix_mlx}.ln_1.weight"
        weight_map[f"{prefix_pt}.self_attn_layer_norm.bias"] = \
            f"{prefix_mlx}.ln_1.bias"
        weight_map[f"{prefix_pt}.final_layer_norm.weight"] = \
            f"{prefix_mlx}.ln_2.weight"
        weight_map[f"{prefix_pt}.final_layer_norm.bias"] = \
            f"{prefix_mlx}.ln_2.bias"

        # Attention (combined QKV)
        weight_map[f"{prefix_pt}.self_attn.q_proj.weight"] = \
            f"{prefix_mlx}.attn.c_attn.weight.q"
        weight_map[f"{prefix_pt}.self_attn.q_proj.bias"] = \
            f"{prefix_mlx}.attn.c_attn.bias.q"
        weight_map[f"{prefix_pt}.self_attn.k_proj.weight"] = \
            f"{prefix_mlx}.attn.c_attn.weight.k"
        weight_map[f"{prefix_pt}.self_attn.k_proj.bias"] = \
            f"{prefix_mlx}.attn.c_attn.bias.k"
        weight_map[f"{prefix_pt}.self_attn.v_proj.weight"] = \
            f"{prefix_mlx}.attn.c_attn.weight.v"
        weight_map[f"{prefix_pt}.self_attn.v_proj.bias"] = \
            f"{prefix_mlx}.attn.c_attn.bias.v"

        # Attention output
        weight_map[f"{prefix_pt}.self_attn.out_proj.weight"] = \
            f"{prefix_mlx}.attn.c_proj.weight"
        weight_map[f"{prefix_pt}.self_attn.out_proj.bias"] = \
            f"{prefix_mlx}.attn.c_proj.bias"

        # MLP
        weight_map[f"{prefix_pt}.fc1.weight"] = \
            f"{prefix_mlx}.mlp.c_fc.weight"
        weight_map[f"{prefix_pt}.fc1.bias"] = \
            f"{prefix_mlx}.mlp.c_fc.bias"
        weight_map[f"{prefix_pt}.fc2.weight"] = \
            f"{prefix_mlx}.mlp.c_proj.weight"
        weight_map[f"{prefix_pt}.fc2.bias"] = \
            f"{prefix_mlx}.mlp.c_proj.bias"

    # Final layer norm
    weight_map["decoder.model.decoder.layer_norm.weight"] = \
        "decoder.decoder.ln_f.weight"
    weight_map["decoder.model.decoder.layer_norm.bias"] = \
        "decoder.decoder.ln_f.bias"

    # LM head
    weight_map["decoder.lm_head.weight"] = \
        "lm_head.weight"

    print(f"Total mappings defined: {len(weight_map)}")

    # Apply weight mapping
    converted_weights = {}
    matched = 0
    missing = []

    for pt_key, mlx_key in weight_map.items():
        if pt_key in pt_weights:
            weight = pt_weights[pt_key]

            # Handle special cases for weight transposition
            if "weight" in pt_key and len(weight.shape) == 2:
                # Linear layers in PyTorch are (out, in), MLX expects (in, out)
                if "embed" not in pt_key.lower():  # Don't transpose embeddings
                    weight = weight.T

            converted_weights[mlx_key] = mx.array(weight)
            matched += 1
        else:
            missing.append(pt_key)

    print(f"✓ Matched {matched} weights")

    if missing:
        print(f"\n⚠️  Missing {len(missing)} weights:")
        for key in missing[:10]:
            print(f"    - {key}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    return converted_weights


def convert_model(input_path, output_path, model_name="microsoft/trocr-base-handwritten"):
    """
    Main conversion function
    """
    print("=" * 70)
    print("TrOCR PyTorch → MLX Converter")
    print("=" * 70)
    print()

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"📥 Loading PyTorch model from: {input_path or model_name}")
    if input_path:
        pt_model = VisionEncoderDecoderModel.from_pretrained(input_path)
        processor = TrOCRProcessor.from_pretrained(input_path)
    else:
        pt_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)

    print("✓ PyTorch model loaded")

    # Get state dict
    pt_state = pt_model.state_dict()
    print(f"✓ State dict loaded: {len(pt_state)} keys")

    # Create MLX model
    print("\n📦 Creating MLX model...")
    mlx_model = create_trocr_base()
    print("✓ MLX model created")

    # Convert weights
    print("\n🔄 Converting weights...")
    converted_weights = map_weights(pt_state, mlx_model)

    # Update MLX model with converted weights
    print("\n📝 Updating MLX model with converted weights...")
    mlx_model.update(converted_weights)
    print("✓ Model updated")

    # Save MLX model
    print(f"\n💾 Saving MLX model to: {output_path}")

    # Save weights as npz
    weights_file = output_path / "weights.npz"
    weight_arrays = {k: np.array(v) for k, v in converted_weights.items()}
    np.savez(weights_file, **weight_arrays)
    print(f"✓ Weights saved: {weights_file}")

    # Save processor (tokenizer)
    processor_path = output_path / "processor"
    processor.save_pretrained(processor_path)
    print(f"✓ Processor saved: {processor_path}")

    # Save config
    config = {
        "model_type": "trocr",
        "encoder": {
            "image_size": 384,
            "patch_size": 16,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
        },
        "decoder": {
            "vocab_size": 50265,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "max_length": 512,
        }
    }

    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_file}")

    print()
    print("=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    print(f"\nMLX model saved to: {output_path}")
    print("\nFiles created:")
    print(f"  - weights.npz      (model weights)")
    print(f"  - config.json      (model configuration)")
    print(f"  - processor/       (tokenizer and feature extractor)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert TrOCR from PyTorch to MLX")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to PyTorch model (or use default HuggingFace model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mlx/trocr-base-handwritten",
        help="Output path for MLX model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="HuggingFace model name (if --input not provided)"
    )

    args = parser.parse_args()

    convert_model(args.input, args.output, args.model_name)


if __name__ == "__main__":
    main()
