#!/usr/bin/env python3
"""
MLX TrOCR Inference Script
Perform OCR using MLX-converted TrOCR model

Author: Diego Alarcon
Date: October 2025
"""

import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor
from mlx_models import create_trocr_base


class MLXTrOCRInference:
    """
    MLX TrOCR Inference wrapper
    """
    def __init__(self, model_path):
        """
        Load MLX TrOCR model and processor

        Args:
            model_path: Path to MLX model directory
        """
        self.model_path = Path(model_path)

        print("=" * 70)
        print("MLX TrOCR Inference")
        print("=" * 70)
        print()

        # Load config
        print(f"📋 Loading config from: {self.model_path}")
        config_file = self.model_path / "config.json"
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        print("✓ Config loaded")

        # Create model
        print("\n📦 Creating MLX model...")
        self.model = create_trocr_base()
        print("✓ Model created")

        # Load weights
        print(f"\n💾 Loading weights from: {self.model_path}")
        weights_file = self.model_path / "weights.npz"

        if not weights_file.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        weights = np.load(weights_file)
        mlx_weights = {k: mx.array(v) for k, v in weights.items()}
        self.model.update(mlx_weights)
        print(f"✓ Loaded {len(mlx_weights)} weight tensors")

        # Load processor
        print(f"\n🔧 Loading processor...")
        processor_path = self.model_path / "processor"
        self.processor = TrOCRProcessor.from_pretrained(processor_path)
        print("✓ Processor loaded")

        print()
        print("=" * 70)
        print("✓ Model ready for inference")
        print("=" * 70)
        print()

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input

        Args:
            image_path: Path to image file

        Returns:
            MLX array of pixel values
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Use processor to prepare image
        pixel_values = self.processor(image, return_tensors="np").pixel_values

        # Convert to MLX array
        pixel_values = mx.array(pixel_values)

        return pixel_values, image

    def generate_text(self, pixel_values, max_length=64, temperature=1.0):
        """
        Generate text from image

        Args:
            pixel_values: Preprocessed image tensor
            max_length: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            pixel_values,
            max_length=max_length,
            temperature=temperature
        )

    def decode_tokens(self, token_ids):
        """
        Decode token IDs to text

        Args:
            token_ids: MLX array of token IDs

        Returns:
            Decoded text string
        """
        # Convert to numpy for tokenizer
        token_ids_np = np.array(token_ids)

        # Decode
        text = self.processor.batch_decode(
            token_ids_np,
            skip_special_tokens=True
        )[0]

        return text

    def process_image(self, image_path, max_length=64, temperature=1.0):
        """
        Complete OCR pipeline: image → text

        Args:
            image_path: Path to image file
            max_length: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            Extracted text
        """
        print(f"🖼️  Processing image: {image_path}")

        # Preprocess
        pixel_values, original_image = self.preprocess_image(image_path)
        print(f"   Image size: {original_image.size}")
        print(f"   Tensor shape: {pixel_values.shape}")

        # Generate
        print(f"   Generating text (max_length={max_length})...")
        token_ids = self.generate_text(
            pixel_values,
            max_length=max_length,
            temperature=temperature
        )
        print(f"   Generated {token_ids.shape[1]} tokens")

        # Decode
        text = self.decode_tokens(token_ids)

        print(f"   ✓ Extracted text: '{text}'")
        print()

        return text


def main():
    parser = argparse.ArgumentParser(description="MLX TrOCR Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="models/mlx/trocr-base-handwritten",
        help="Path to MLX model directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Create inference instance
    inference = MLXTrOCRInference(args.model)

    # Process image
    text = inference.process_image(
        args.image,
        max_length=args.max_length,
        temperature=args.temperature
    )

    # Output
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Text:  {text}")
    print("=" * 70)


if __name__ == "__main__":
    main()
