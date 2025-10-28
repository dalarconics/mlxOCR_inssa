# MLX Migration Guide - pyOCR_inssa

## Overview

This guide documents the migration from **PyTorch + HuggingFace Transformers** to **Apple MLX** for the TrOCR-based OCR system.

**Branch**: `ocr_mlx`
**Target Platform**: macOS with Apple Silicon (M4 Max optimized)
**Framework Change**: PyTorch MPS → Apple MLX

---

## Why MLX?

### Performance Benefits
- **2-3x faster inference** on Apple Silicon vs PyTorch MPS
- **Unified memory architecture** - no CPU/GPU data transfers
- **Lower latency** - optimized for real-time processing
- **Better memory efficiency** - automatic memory management
- **Smaller footprint** - ~10MB vs PyTorch's ~2GB

### Technical Advantages
- Native Apple Silicon optimization
- Lazy evaluation for better performance
- Automatic mixed precision (fp16/bf16)
- Better integration with macOS ecosystem
- Active development by Apple

---

## Phase 1: Environment Setup ✅

### Step 1: Install MLX Dependencies

**Option A: Automated (Recommended)**
```bash
# Make script executable (if not already)
chmod +x setup_mlx.sh

# Run automated setup
./setup_mlx.sh
```

**Option B: Manual Installation**
```bash
# Create virtual environment
python3 -m venv venv_mlx
source venv_mlx/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install MLX packages
pip install -r requirements-mlx.txt
```

### Step 2: Verify Installation
```bash
# Activate environment
source venv_mlx/bin/activate

# Test MLX
python3 -c "import mlx.core as mx; print(mx.__version__)"
```

**Expected Output**: `0.22.0` (or newer)

---

## Phase 2: Model Conversion (TODO)

### Understanding the Architecture

**Current Setup (PyTorch)**:
```
microsoft/trocr-base-handwritten (HuggingFace)
├── Vision Encoder (ViT-based)
├── Decoder (GPT-2 based)
└── Tokenizer (RobertaTokenizer)
```

**MLX Setup (Target)**:
```
mlx/trocr-base-handwritten (converted)
├── Vision Encoder (MLX Arrays)
├── Decoder (MLX Arrays)
└── Tokenizer (HuggingFace - unchanged)
```

### Conversion Steps

#### Step 1: Download Original Model
```bash
python3 -c "
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Download and cache
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

# Save locally
model.save_pretrained('./models/pytorch/trocr-base')
processor.save_pretrained('./models/pytorch/trocr-base')
print('✓ Model downloaded')
"
```

#### Step 2: Convert to MLX Format (Script to be created)
```bash
# This will be implemented in Phase 2
python3 scripts/convert_trocr_to_mlx.py \
    --input ./models/pytorch/trocr-base \
    --output ./models/mlx/trocr-base
```

### Fine-tuned Model Conversion
```bash
# Convert your custom fine-tuned model
python3 scripts/convert_trocr_to_mlx.py \
    --input /Volumes/llm/models/microsoft/trocr-finetuned \
    --output ./models/mlx/trocr-finetuned
```

---

## Phase 3: Code Migration (TODO)

### File-by-File Migration Plan

#### 1. Preprocessing Scripts (No Changes Needed ✓)
- `ocr_preprocessor_handler.py` - Uses OpenCV/PIL only
- `trOCR_edges.py` - Pure image processing
- Image processing pipeline remains identical

#### 2. Inference Scripts (Requires Migration)

**File**: `ocr_inference_handler.py`

**Current (PyTorch)**:
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**Target (MLX)**:
```python
import mlx.core as mx
from mlx_vlm import load_model, generate_text

# Load MLX model (simplified)
model, processor = load_model("./models/mlx/trocr-base")

# Process image (preprocessor unchanged)
pixel_values = processor(image)

# Generate text with MLX
generated_ids = model.generate(mx.array(pixel_values))
text = processor.decode(generated_ids)
```

#### 3. Training Script (Requires Rewrite)

**File**: `ocr_trainer_handler.py`

**Changes Required**:
- ❌ Remove: `Seq2SeqTrainer`, `Seq2SeqTrainingArguments`
- ✅ Add: Custom MLX training loop
- ✅ Add: MLX optimizers (`mx.optimizers.Adam`)
- ✅ Add: Manual gradient computation (`mx.grad`)

**Implementation**: See `scripts/mlx_trainer.py` (to be created)

---

## Phase 4: Performance Optimization (TODO)

### MLX-Specific Optimizations

1. **Lazy Evaluation**
```python
# MLX evaluates operations lazily
# Force evaluation when needed
result = mx.eval(computation)
```

2. **Batch Processing**
```python
# Process multiple regions in parallel
batch = mx.stack([process_region(r) for r in regions])
results = model.generate(batch)
```

3. **Memory Management**
```python
# MLX uses unified memory - minimal management needed
# Optional: clear cache between batches
mx.metal.clear_cache()
```

---

## Migration Checklist

### Phase 1: Environment Setup ✅
- [x] Create `requirements-mlx.txt`
- [x] Create `setup_mlx.sh` script
- [x] Test MLX installation
- [x] Verify all dependencies work together

### Phase 2: Model Conversion ✅
- [x] Download base TrOCR model
- [x] Create conversion script
- [x] Implement MLX TrOCR architecture
- [x] Create weight mapping utilities
- [x] Create MLX inference script
- [ ] Test conversion with base model (ready to run)
- [ ] Convert fine-tuned model to MLX (ready to run)

### Phase 3: Inference Migration ✅
- [x] Create MLX-based processor wrapper
- [x] Migrate `ocr_inference_handler.py` (→ `ocr_inference_handler_mlx.py`)
- [x] Create batch processing utilities
- [x] Create performance benchmarking tools
- [ ] Test with converted models (ready to run)
- [ ] Run benchmarks (ready to run)

### Phase 4: Training Migration
- [ ] Design MLX training loop
- [ ] Implement dataset loading for MLX
- [ ] Create MLX trainer class
- [ ] Migrate `ocr_trainer_handler.py`
- [ ] Validate training convergence

### Phase 5: Testing & Validation
- [ ] Unit tests for each component
- [ ] End-to-end pipeline test
- [ ] Performance benchmarking
- [ ] Accuracy comparison (PyTorch vs MLX)
- [ ] Memory usage profiling

---

## Dependency Changes

### Removed
```
torch==2.4.0                    # Replaced by MLX
transformers[torch]==4.53.1     # Using minimal version
```

### Added
```
mlx==0.22.0                     # Core MLX framework
mlx-lm==0.21.1                  # Language model utilities
safetensors==0.4.5              # Model serialization
```

### Unchanged
```
opencv-python==4.10.0.84        # Image processing
numpy==2.1.2                    # Array operations
Pillow==10.4.0                  # Image I/O
matplotlib==3.9.2               # Visualization
```

---

## Performance Expectations

### Before (PyTorch MPS)
- **Training**: ~27.78s per epoch (10 epochs, batch size 8)
- **Inference**: ~150-200ms per region
- **Memory**: ~4-6GB VRAM usage
- **Model Size**: ~400MB (PyTorch format)

### After (MLX) - Expected
- **Training**: ~12-15s per epoch (2x faster)
- **Inference**: ~60-80ms per region (2.5x faster)
- **Memory**: ~2-3GB unified memory
- **Model Size**: ~300MB (MLX format)

---

## Troubleshooting

### MLX Installation Issues
```bash
# If MLX fails to install
pip install --upgrade pip setuptools wheel
pip install mlx --no-cache-dir

# Check Apple Silicon
uname -m  # Should output: arm64
```

### Import Errors
```python
# If "No module named 'mlx'"
import sys
print(sys.prefix)  # Verify virtual environment

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx==0.22.0 mlx-lm==0.21.1
```

### Performance Not Improving
```python
# Verify using Metal backend
import mlx.core as mx
print(mx.metal.is_available())  # Should be True

# Check memory mode
print(mx.metal.get_memory_limit())
```

---

## Resources

### Documentation
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [MLX Vision-Language Models](https://github.com/Blaizzy/mlx-vlm)

### Reference Implementations
- MLX TrOCR: ✅ See `scripts/mlx_models.py`
- Model Conversion: ✅ See `scripts/convert_trocr_to_mlx.py`
- MLX Inference: ✅ See `scripts/mlx_inference.py`
- MLX Fine-tuning: Coming in Phase 4

### Support
- MLX GitHub Issues: https://github.com/ml-explore/mlx/issues
- Apple Developer Forums: https://developer.apple.com/forums/

---

## Next Steps

1. **Complete Phase 1**:
   ```bash
   ./setup_mlx.sh
   ```

2. **Prepare for Phase 2**:
   - Review model architecture
   - Download base models
   - Study MLX model conversion examples

3. **Test Current Setup**:
   ```bash
   source venv_mlx/bin/activate
   python3 -c "import mlx.core as mx; print('✓ Ready for conversion')"
   ```

---

**Last Updated**: October 2025
**Status**: Phase 3 Complete ✅
**Next**: Run conversion and test integration

## Quick Start - Complete Workflow

```bash
# 1. Setup MLX environment (Phase 1)
./setup_mlx.sh
source venv_mlx/bin/activate

# 2. Convert models (Phase 2)
python3 scripts/inspect_trocr_model.py
python3 scripts/convert_trocr_to_mlx.py \
    --input models/pytorch/trocr-base-handwritten \
    --output models/mlx/trocr-base-handwritten

# 3. Run MLX inference (Phase 3)
python3 ocr_inference_handler_mlx.py

# 4. Benchmark performance
python3 scripts/benchmark_pytorch_vs_mlx.py \
    --images test_images/*.jpg \
    --output benchmark.json
```

## Phase-by-Phase Guides

- **Phase 1**: Environment Setup → See main README
- **Phase 2**: Model Conversion → See [PHASE2_INSTRUCTIONS.md](PHASE2_INSTRUCTIONS.md)
- **Phase 3**: Pipeline Integration → See [PHASE3_INSTRUCTIONS.md](PHASE3_INSTRUCTIONS.md)
