#!/usr/bin/env python3
"""
Inspect TrOCR Model Architecture
Analyzes the structure of microsoft/trocr-base-handwritten for MLX conversion

Author: Diego Alarcon
Date: October 2025
"""

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import json
from pathlib import Path

def inspect_model():
    """Inspect TrOCR model architecture and save analysis"""

    print("=" * 70)
    print("TrOCR Model Architecture Inspector")
    print("=" * 70)
    print()

    # Load model and processor
    print("📥 Loading model from HuggingFace...")
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    print(f"✓ Model loaded: {model_name}")
    print()

    # Basic info
    print("=" * 70)
    print("MODEL OVERVIEW")
    print("=" * 70)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Encoder class: {model.encoder.__class__.__name__}")
    print(f"Decoder class: {model.decoder.__class__.__name__}")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("=" * 70)
    print("PARAMETER COUNT")
    print("=" * 70)
    print(f"Total parameters:   {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print()

    # Model configuration
    print("=" * 70)
    print("ENCODER CONFIG (ViT)")
    print("=" * 70)
    enc_config = model.config.encoder
    print(f"Hidden size:        {enc_config.hidden_size}")
    print(f"Num hidden layers:  {enc_config.num_hidden_layers}")
    print(f"Num attention heads:{enc_config.num_attention_heads}")
    print(f"Image size:         {enc_config.image_size}")
    print(f"Patch size:         {enc_config.patch_size}")
    print()

    print("=" * 70)
    print("DECODER CONFIG (GPT-2)")
    print("=" * 70)
    dec_config = model.config.decoder
    print(f"Hidden size:        {dec_config.n_embd}")
    print(f"Num layers:         {dec_config.n_layer}")
    print(f"Num attention heads:{dec_config.n_head}")
    print(f"Vocab size:         {dec_config.vocab_size}")
    print()

    # Processor info
    print("=" * 70)
    print("PROCESSOR INFO")
    print("=" * 70)
    print(f"Feature extractor:  {processor.feature_extractor.__class__.__name__}")
    print(f"Tokenizer:          {processor.tokenizer.__class__.__name__}")
    print(f"Image mean:         {processor.feature_extractor.image_mean}")
    print(f"Image std:          {processor.feature_extractor.image_std}")
    print()

    # Layer breakdown
    print("=" * 70)
    print("ENCODER LAYERS")
    print("=" * 70)
    for name, module in model.encoder.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    print()

    print("=" * 70)
    print("DECODER LAYERS")
    print("=" * 70)
    for name, module in model.decoder.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    print()

    # State dict analysis
    print("=" * 70)
    print("STATE DICT KEYS (first 20)")
    print("=" * 70)
    state_dict = model.state_dict()
    for i, key in enumerate(list(state_dict.keys())[:20]):
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
    print(f"  ... and {len(state_dict) - 20} more keys")
    print()

    # Save model locally for conversion
    print("=" * 70)
    print("SAVING MODEL LOCALLY")
    print("=" * 70)
    output_dir = Path("models/pytorch/trocr-base-handwritten")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"✓ Model saved to: {output_dir}")
    print()

    # Save architecture info as JSON
    arch_info = {
        "model_name": model_name,
        "total_parameters": total_params,
        "encoder": {
            "class": model.encoder.__class__.__name__,
            "parameters": encoder_params,
            "hidden_size": enc_config.hidden_size,
            "num_layers": enc_config.num_hidden_layers,
            "num_heads": enc_config.num_attention_heads,
            "image_size": enc_config.image_size,
            "patch_size": enc_config.patch_size,
        },
        "decoder": {
            "class": model.decoder.__class__.__name__,
            "parameters": decoder_params,
            "hidden_size": dec_config.n_embd,
            "num_layers": dec_config.n_layer,
            "num_heads": dec_config.n_head,
            "vocab_size": dec_config.vocab_size,
        },
        "processor": {
            "image_mean": processor.feature_extractor.image_mean,
            "image_std": processor.feature_extractor.image_std,
            "size": processor.feature_extractor.size,
        }
    }

    arch_file = output_dir / "architecture_info.json"
    with open(arch_file, 'w') as f:
        json.dump(arch_info, f, indent=2)

    print(f"✓ Architecture info saved to: {arch_file}")
    print()

    # Test inference
    print("=" * 70)
    print("TEST INFERENCE")
    print("=" * 70)
    from PIL import Image
    import requests

    # Create a simple test image
    test_img = Image.new('RGB', (384, 384), color='white')

    print("Testing with dummy image...")
    pixel_values = processor(test_img, return_tensors="pt").pixel_values
    print(f"  Input shape: {pixel_values.shape}")

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=16)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"  Output shape: {generated_ids.shape}")
    print(f"  Generated text: '{text}'")
    print()

    print("=" * 70)
    print("✓ Inspection complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review architecture_info.json")
    print("  2. Create MLX model classes based on this structure")
    print("  3. Run conversion script to convert weights")
    print()

if __name__ == "__main__":
    inspect_model()
