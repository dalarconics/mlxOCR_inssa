#!/usr/bin/env python3
"""
PyTorch vs MLX Performance Benchmark
Compare inference speed, memory usage, and accuracy between PyTorch MPS and MLX

Author: Diego Alarcon
Date: October 2025
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import json

# MLX imports
sys.path.insert(0, str(Path(__file__).parent))
from mlx_processor import load_mlx_processor
import mlx.core as mx

# PyTorch imports
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will only benchmark MLX")


class PerformanceBenchmark:
    """Benchmark framework for comparing PyTorch and MLX"""

    def __init__(
        self,
        pytorch_model: str = "microsoft/trocr-base-handwritten",
        mlx_model: str = "models/mlx/trocr-base-handwritten"
    ):
        """
        Initialize benchmark

        Args:
            pytorch_model: PyTorch model name or path
            mlx_model: MLX model path
        """
        self.pytorch_model_name = pytorch_model
        self.mlx_model_path = mlx_model

        self.results = {
            'pytorch': {'enabled': PYTORCH_AVAILABLE},
            'mlx': {'enabled': True}
        }

    def load_pytorch_model(self):
        """Load PyTorch model"""
        if not PYTORCH_AVAILABLE:
            return None, None

        print("📦 Loading PyTorch model...")
        start = time.time()

        processor = TrOCRProcessor.from_pretrained(self.pytorch_model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.pytorch_model_name)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)

        load_time = time.time() - start

        self.results['pytorch']['load_time'] = load_time
        self.results['pytorch']['device'] = device

        print(f"   ✓ Loaded in {load_time:.2f}s on {device}")

        return processor, model

    def load_mlx_model(self):
        """Load MLX model"""
        print("📦 Loading MLX model...")
        start = time.time()

        processor = load_mlx_processor(self.mlx_model_path)

        load_time = time.time() - start

        self.results['mlx']['load_time'] = load_time
        self.results['mlx']['device'] = 'mlx_unified'

        print(f"   ✓ Loaded in {load_time:.2f}s on MLX")

        return processor

    def benchmark_pytorch(
        self,
        images: List[Image.Image],
        processor,
        model,
        max_length: int = 64
    ) -> Dict:
        """Benchmark PyTorch inference"""
        if not PYTORCH_AVAILABLE:
            return {}

        print("\n⚡ Benchmarking PyTorch...")

        device = self.results['pytorch']['device']
        times = []
        texts = []

        # Warmup
        pixel_values = processor(images[0], return_tensors="pt").pixel_values.to(device)
        _ = model.generate(pixel_values, max_length=max_length)

        # Actual benchmark
        for idx, image in enumerate(images):
            start = time.time()

            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_length=max_length)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            elapsed = time.time() - start

            times.append(elapsed)
            texts.append(text)

            print(f"   Image {idx+1}/{len(images)}: {elapsed*1000:.1f}ms - '{text}'")

        results = {
            'times': times,
            'texts': texts,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': len(images) / sum(times)
        }

        return results

    def benchmark_mlx(
        self,
        images: List[Image.Image],
        processor,
        max_length: int = 64
    ) -> Dict:
        """Benchmark MLX inference"""
        print("\n⚡ Benchmarking MLX...")

        times = []
        texts = []

        # Warmup
        _ = processor.process_image(images[0], max_length=max_length)

        # Actual benchmark
        for idx, image in enumerate(images):
            start = time.time()

            text = processor.process_image(image, max_length=max_length)

            elapsed = time.time() - start

            times.append(elapsed)
            texts.append(text)

            print(f"   Image {idx+1}/{len(images)}: {elapsed*1000:.1f}ms - '{text}'")

        results = {
            'times': times,
            'texts': texts,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': len(images) / sum(times)
        }

        return results

    def compare_accuracy(self, pytorch_results: Dict, mlx_results: Dict) -> Dict:
        """Compare text outputs between PyTorch and MLX"""
        if not pytorch_results:
            return {}

        pytorch_texts = pytorch_results.get('texts', [])
        mlx_texts = mlx_results.get('texts', [])

        if not pytorch_texts or not mlx_texts:
            return {}

        matches = sum(1 for pt, mx in zip(pytorch_texts, mlx_texts) if pt == mx)
        total = len(pytorch_texts)

        return {
            'exact_matches': matches,
            'total': total,
            'match_rate': matches / total if total > 0 else 0,
            'differences': [
                {'idx': i, 'pytorch': pt, 'mlx': mx}
                for i, (pt, mx) in enumerate(zip(pytorch_texts, mlx_texts))
                if pt != mx
            ]
        }

    def print_results(self):
        """Print formatted benchmark results"""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        # Model loading
        print("\n📦 MODEL LOADING")
        print("-" * 70)
        if self.results['pytorch']['enabled']:
            print(f"PyTorch: {self.results['pytorch']['load_time']:.2f}s ({self.results['pytorch']['device']})")
        print(f"MLX:     {self.results['mlx']['load_time']:.2f}s ({self.results['mlx']['device']})")

        if self.results['pytorch']['enabled'] and 'load_time' in self.results['pytorch']:
            speedup = self.results['pytorch']['load_time'] / self.results['mlx']['load_time']
            print(f"\n✓ MLX is {speedup:.2f}x faster at loading")

        # Inference performance
        if 'inference' in self.results['pytorch'] and 'inference' in self.results['mlx']:
            print("\n⚡ INFERENCE PERFORMANCE")
            print("-" * 70)

            pt_inf = self.results['pytorch']['inference']
            mlx_inf = self.results['mlx']['inference']

            print(f"PyTorch:")
            print(f"  Avg time:    {pt_inf['avg_time']*1000:.1f}ms")
            print(f"  Min/Max:     {pt_inf['min_time']*1000:.1f}ms / {pt_inf['max_time']*1000:.1f}ms")
            print(f"  Throughput:  {pt_inf['throughput']:.2f} images/sec")

            print(f"\nMLX:")
            print(f"  Avg time:    {mlx_inf['avg_time']*1000:.1f}ms")
            print(f"  Min/Max:     {mlx_inf['min_time']*1000:.1f}ms / {mlx_inf['max_time']*1000:.1f}ms")
            print(f"  Throughput:  {mlx_inf['throughput']:.2f} images/sec")

            speedup = pt_inf['avg_time'] / mlx_inf['avg_time']
            print(f"\n✓ MLX is {speedup:.2f}x faster than PyTorch")

        # Accuracy comparison
        if 'accuracy' in self.results:
            acc = self.results['accuracy']
            print("\n🎯 ACCURACY COMPARISON")
            print("-" * 70)
            print(f"Exact matches: {acc['exact_matches']}/{acc['total']} ({acc['match_rate']*100:.1f}%)")

            if acc['differences']:
                print(f"\nDifferences found:")
                for diff in acc['differences'][:5]:
                    print(f"  Image {diff['idx']+1}:")
                    print(f"    PyTorch: '{diff['pytorch']}'")
                    print(f"    MLX:     '{diff['mlx']}'")
                if len(acc['differences']) > 5:
                    print(f"  ... and {len(acc['differences'])-5} more")

        print("\n" + "=" * 70)

    def save_results(self, output_path: str):
        """Save results to JSON"""
        output_path = Path(output_path)

        # Convert numpy types to native Python
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        results_clean = convert_types(self.results)

        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")

    def run(self, image_paths: List[str], max_length: int = 64, save_path: str = None):
        """Run complete benchmark"""
        print("=" * 70)
        print("PyTorch vs MLX Benchmark")
        print("=" * 70)
        print(f"\nImages to process: {len(image_paths)}")
        print()

        # Load images
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Load models
        if PYTORCH_AVAILABLE:
            pt_processor, pt_model = self.load_pytorch_model()
        else:
            pt_processor, pt_model = None, None

        mlx_processor = self.load_mlx_model()

        # Run benchmarks
        if pt_processor and pt_model:
            self.results['pytorch']['inference'] = self.benchmark_pytorch(
                images, pt_processor, pt_model, max_length
            )

        self.results['mlx']['inference'] = self.benchmark_mlx(
            images, mlx_processor, max_length
        )

        # Compare accuracy
        if pt_processor and pt_model:
            self.results['accuracy'] = self.compare_accuracy(
                self.results['pytorch'].get('inference', {}),
                self.results['mlx']['inference']
            )

        # Print results
        self.print_results()

        # Save if requested
        if save_path:
            self.save_results(save_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs MLX for TrOCR")

    parser.add_argument(
        "--images",
        nargs='+',
        required=True,
        help="List of image paths to benchmark"
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="PyTorch model name or path"
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="models/mlx/trocr-base-handwritten",
        help="MLX model path"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = PerformanceBenchmark(
        pytorch_model=args.pytorch_model,
        mlx_model=args.mlx_model
    )

    benchmark.run(
        image_paths=args.images,
        max_length=args.max_length,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
