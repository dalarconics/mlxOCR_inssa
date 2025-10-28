#!/usr/bin/env python3
"""
MLX Batch OCR Processor
Efficiently process multiple images or documents using MLX TrOCR

Features:
- Parallel region processing
- Progress tracking
- CSV/JSON export
- Performance metrics

Author: Diego Alarcon
Date: October 2025
"""

import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import mlx.core as mx
from mlx_processor import load_mlx_processor
from PIL import Image


class MLXBatchProcessor:
    """Batch processor for MLX TrOCR"""

    def __init__(self, model_path: str, verbose: bool = True):
        """
        Initialize batch processor

        Args:
            model_path: Path to MLX model
            verbose: Print progress information
        """
        self.verbose = verbose
        self.processor = load_mlx_processor(model_path)
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0
        }

    def process_single(
        self,
        image_path: Union[str, Path],
        max_length: int = 64,
        temperature: float = 1.0
    ) -> Dict:
        """
        Process a single image

        Args:
            image_path: Path to image
            max_length: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            Dict with result and metadata
        """
        image_path = Path(image_path)
        start_time = time.time()

        try:
            # Process image
            text = self.processor.process_image(
                str(image_path),
                max_length=max_length,
                temperature=temperature
            )

            processing_time = time.time() - start_time

            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'text': text,
                'success': True,
                'processing_time': processing_time,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }

            self.stats['successful'] += 1

        except Exception as e:
            processing_time = time.time() - start_time

            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'text': None,
                'success': False,
                'processing_time': processing_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            self.stats['failed'] += 1

        self.stats['total_time'] += processing_time
        self.stats['total_images'] += 1

        return result

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        max_length: int = 64,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process multiple images

        Args:
            image_paths: List of image paths
            max_length: Maximum sequence length
            temperature: Sampling temperature
            show_progress: Show progress information

        Returns:
            List of result dictionaries
        """
        results = []
        total = len(image_paths)

        if self.verbose:
            print("=" * 70)
            print(f"MLX BATCH PROCESSING")
            print("=" * 70)
            print(f"Total images: {total}")
            print()

        for idx, image_path in enumerate(image_paths):
            if show_progress and self.verbose:
                print(f"[{idx+1}/{total}] Processing: {Path(image_path).name}")

            result = self.process_single(
                image_path,
                max_length=max_length,
                temperature=temperature
            )

            results.append(result)

            if show_progress and self.verbose and result['success']:
                print(f"   ✓ Text: '{result['text']}' ({result['processing_time']:.3f}s)")
            elif show_progress and self.verbose:
                print(f"   ✗ Error: {result['error']}")

        # Calculate statistics
        if self.stats['total_images'] > 0:
            self.stats['avg_time_per_image'] = (
                self.stats['total_time'] / self.stats['total_images']
            )

        if self.verbose:
            self._print_stats()

        return results

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        recursive: bool = False,
        max_length: int = 64,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Process all images in a directory

        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.jpg", "*.png")
            recursive: Search recursively
            max_length: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            List of result dictionaries
        """
        directory = Path(directory)

        if recursive:
            image_paths = list(directory.rglob(pattern))
        else:
            image_paths = list(directory.glob(pattern))

        if self.verbose:
            print(f"Found {len(image_paths)} images matching '{pattern}' in {directory}")
            print()

        return self.process_batch(
            image_paths,
            max_length=max_length,
            temperature=temperature
        )

    def export_results_csv(self, results: List[Dict], output_path: Union[str, Path]):
        """Export results to CSV"""
        output_path = Path(output_path)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image_name', 'image_path', 'text', 'success',
                'processing_time', 'error', 'timestamp'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        if self.verbose:
            print(f"✓ Results exported to CSV: {output_path}")

    def export_results_json(self, results: List[Dict], output_path: Union[str, Path]):
        """Export results to JSON"""
        output_path = Path(output_path)

        output = {
            'metadata': {
                'total_images': self.stats['total_images'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'total_time': self.stats['total_time'],
                'avg_time_per_image': self.stats['avg_time_per_image'],
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"✓ Results exported to JSON: {output_path}")

    def _print_stats(self):
        """Print processing statistics"""
        print()
        print("=" * 70)
        print("PROCESSING STATISTICS")
        print("=" * 70)
        print(f"Total images:          {self.stats['total_images']}")
        print(f"Successful:            {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Total time:            {self.stats['total_time']:.2f}s")
        print(f"Avg time per image:    {self.stats['avg_time_per_image']:.3f}s")
        if self.stats['successful'] > 0:
            throughput = self.stats['successful'] / self.stats['total_time']
            print(f"Throughput:            {throughput:.2f} images/second")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MLX Batch OCR Processor")

    parser.add_argument(
        "--model",
        type=str,
        default="models/mlx/trocr-base-handwritten",
        help="Path to MLX model"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--images",
        nargs='+',
        help="List of image paths"
    )
    input_group.add_argument(
        "--directory",
        type=str,
        help="Directory containing images"
    )

    # Directory options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="File pattern for directory mode (default: *.jpg)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory recursively"
    )

    # OCR options
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

    # Output options
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Export results to JSON file"
    )

    # Misc
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Create processor
    processor = MLXBatchProcessor(args.model, verbose=not args.quiet)

    # Process images
    if args.images:
        results = processor.process_batch(
            args.images,
            max_length=args.max_length,
            temperature=args.temperature
        )
    else:
        results = processor.process_directory(
            args.directory,
            pattern=args.pattern,
            recursive=args.recursive,
            max_length=args.max_length,
            temperature=args.temperature
        )

    # Export results
    if args.output_csv:
        processor.export_results_csv(results, args.output_csv)

    if args.output_json:
        processor.export_results_json(results, args.output_json)

    # Print summary if no export specified
    if not args.output_csv and not args.output_json and not args.quiet:
        print()
        print("EXTRACTED TEXTS")
        print("=" * 70)
        for result in results:
            if result['success']:
                print(f"{result['image_name']}: {result['text']}")


if __name__ == "__main__":
    main()
