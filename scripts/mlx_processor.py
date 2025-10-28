#!/usr/bin/env python3
"""
MLX TrOCR Processor
Drop-in replacement for HuggingFace TrOCRProcessor using MLX backend

This module provides a compatible interface with the existing codebase
while using MLX for inference on Apple Silicon.

Author: Diego Alarcon
Date: October 2025
"""

import json
from pathlib import Path
from typing import Union, List, Optional
import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor
from mlx_models import create_trocr_base


class MLXTrOCRProcessor:
    """
    MLX-based TrOCR processor
    Compatible with existing pipeline, optimized for Apple Silicon
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize MLX TrOCR processor

        Args:
            model_path: Path to MLX model directory
            device: Device to use ('auto' uses MLX automatically)
        """
        self.model_path = Path(model_path)
        self.device = device

        print(f"🚀 Initializing MLX TrOCR Processor")
        print(f"   Model path: {self.model_path}")

        # Load config
        config_file = self.model_path / "config.json"
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Create and load MLX model
        self.model = create_trocr_base()

        # Load weights
        weights_file = self.model_path / "weights.npz"
        if weights_file.exists():
            weights = np.load(weights_file)
            mlx_weights = {k: mx.array(v) for k, v in weights.items()}
            self.model.update(mlx_weights)
            print(f"   ✓ Loaded {len(mlx_weights)} weight tensors")
        else:
            print(f"   ⚠️  No weights found at {weights_file}")

        # Load processor (for tokenizer and feature extractor)
        processor_path = self.model_path / "processor"
        self.processor = TrOCRProcessor.from_pretrained(processor_path)
        print(f"   ✓ Processor loaded")

        print(f"   ✓ MLX TrOCR ready (using unified memory)")

    def __call__(self, images, return_tensors: str = "mlx", **kwargs):
        """
        Process images for model input

        Args:
            images: PIL Image or list of images
            return_tensors: Format to return ('mlx', 'np', or 'pt')

        Returns:
            Processed pixel values
        """
        # Use HuggingFace processor for image preprocessing
        pixel_values = self.processor(
            images,
            return_tensors="np"
        ).pixel_values

        # Convert to MLX if needed
        if return_tensors == "mlx":
            return {"pixel_values": mx.array(pixel_values)}
        elif return_tensors == "np":
            return {"pixel_values": pixel_values}
        else:
            return {"pixel_values": pixel_values}

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        """
        Decode token sequences to text

        Args:
            sequences: Token IDs (MLX array or numpy array)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        # Convert MLX array to numpy if needed
        if isinstance(sequences, mx.array):
            sequences = np.array(sequences)

        # Use HuggingFace tokenizer for decoding
        return self.processor.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens
        )

    def generate(
        self,
        pixel_values,
        max_length: int = 64,
        temperature: float = 1.0,
        return_dict: bool = False
    ):
        """
        Generate text from images

        Args:
            pixel_values: Preprocessed image tensor
            max_length: Maximum sequence length
            temperature: Sampling temperature
            return_dict: Whether to return as dict

        Returns:
            Generated token IDs
        """
        # Convert to MLX array if needed
        if isinstance(pixel_values, np.ndarray):
            pixel_values = mx.array(pixel_values)

        # Generate using MLX model
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            temperature=temperature
        )

        if return_dict:
            return {"sequences": generated_ids}
        return generated_ids

    def process_image(
        self,
        image: Union[str, Image.Image],
        max_length: int = 64,
        temperature: float = 1.0
    ) -> str:
        """
        Complete OCR pipeline: image → text

        Args:
            image: Image path or PIL Image
            max_length: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            Extracted text
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Preprocess
        inputs = self(image, return_tensors="mlx")
        pixel_values = inputs["pixel_values"]

        # Generate
        generated_ids = self.generate(
            pixel_values,
            max_length=max_length,
            temperature=temperature
        )

        # Decode
        text = self.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()

    def process_batch(
        self,
        images: List[Union[str, Image.Image]],
        max_length: int = 64,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> List[str]:
        """
        Process multiple images in batch

        Args:
            images: List of image paths or PIL Images
            max_length: Maximum sequence length
            temperature: Sampling temperature
            show_progress: Whether to show progress

        Returns:
            List of extracted texts
        """
        results = []

        # Load images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        # Process in batches
        batch_size = 8  # Adjust based on memory
        num_batches = (len(pil_images) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(pil_images))
            batch = pil_images[start_idx:end_idx]

            if show_progress:
                print(f"   Processing batch {i+1}/{num_batches} ({len(batch)} images)")

            # Preprocess batch
            inputs = self(batch, return_tensors="mlx")
            pixel_values = inputs["pixel_values"]

            # Generate for batch
            generated_ids = self.generate(
                pixel_values,
                max_length=max_length,
                temperature=temperature
            )

            # Decode batch
            texts = self.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend([t.strip() for t in texts])

        return results


def load_mlx_processor(
    model_path: str = "models/mlx/trocr-base-handwritten",
    device: str = "auto"
) -> MLXTrOCRProcessor:
    """
    Convenience function to load MLX processor

    Args:
        model_path: Path to MLX model
        device: Device to use

    Returns:
        MLXTrOCRProcessor instance
    """
    return MLXTrOCRProcessor(model_path, device)


if __name__ == "__main__":
    # Test the processor
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mlx_processor.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/mlx/trocr-base-handwritten"

    print("=" * 70)
    print("MLX TrOCR Processor Test")
    print("=" * 70)
    print()

    # Load processor
    processor = load_mlx_processor(model_path)

    # Process image
    print(f"\n📸 Processing image: {image_path}")
    text = processor.process_image(image_path)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Text: {text}")
    print("=" * 70)
