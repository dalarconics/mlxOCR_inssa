# Phase 2: Model Conversion - Step-by-Step Instructions

## Overview

Phase 2 converts the HuggingFace TrOCR model from PyTorch format to MLX format for optimal performance on Apple Silicon.

## Files Created

### 1. Model Architecture
- **`scripts/mlx_models.py`** - Complete MLX implementation of TrOCR
  - Vision Transformer (ViT) encoder
  - GPT-2 decoder
  - Text generation logic

### 2. Conversion Tools
- **`scripts/inspect_trocr_model.py`** - Analyze PyTorch model structure
- **`scripts/convert_trocr_to_mlx.py`** - Convert weights to MLX format
- **`scripts/mlx_inference.py`** - Run inference with MLX model

---

## Step-by-Step Execution

### Step 1: Inspect Original Model

First, let's understand the PyTorch model structure:

```bash
# Activate MLX environment
source venv_mlx/bin/activate

# Inspect the TrOCR model
cd /Users/alarcondiegof/Documentos/CodeProjects/pyOCR_inssa
python3 scripts/inspect_trocr_model.py
```

**What this does**:
- Downloads microsoft/trocr-base-handwritten from HuggingFace
- Analyzes architecture (encoder/decoder structure)
- Saves model locally to `models/pytorch/trocr-base-handwritten/`
- Creates `architecture_info.json` with model specs
- Tests basic inference

**Expected output**: Model saved to `models/pytorch/trocr-base-handwritten/`

---

### Step 2: Convert Model to MLX

Convert the PyTorch model to MLX format:

```bash
# Convert base model
python3 scripts/convert_trocr_to_mlx.py \
    --input models/pytorch/trocr-base-handwritten \
    --output models/mlx/trocr-base-handwritten
```

**What this does**:
- Loads PyTorch model weights
- Maps weights to MLX model structure
- Converts weight format (PyTorch → MLX)
- Saves converted model to MLX format
- Preserves tokenizer and processor

**Expected output**:
```
models/mlx/trocr-base-handwritten/
├── weights.npz         # MLX model weights
├── config.json         # Model configuration
└── processor/          # Tokenizer & feature extractor
```

---

### Step 3: Test MLX Inference

Test the converted model with a sample image:

```bash
# Test with a sample image
python3 scripts/mlx_inference.py \
    --model models/mlx/trocr-base-handwritten \
    --image path/to/your/test/image.jpg \
    --max-length 64
```

**What this does**:
- Loads MLX model and weights
- Preprocesses input image
- Runs inference using MLX
- Decodes output to text

---

### Step 4: Convert Fine-tuned Model (Optional)

If you have a fine-tuned model, convert it too:

```bash
# Convert your fine-tuned model
python3 scripts/convert_trocr_to_mlx.py \
    --input /Volumes/llm/models/microsoft/trocr-finetuned \
    --output models/mlx/trocr-finetuned
```

---

## Verification Checklist

After completing the steps above, verify:

- [ ] Base model downloaded to `models/pytorch/trocr-base-handwritten/`
- [ ] MLX model created in `models/mlx/trocr-base-handwritten/`
- [ ] `weights.npz` file exists and is ~300MB
- [ ] `config.json` contains correct model configuration
- [ ] `processor/` directory contains tokenizer files
- [ ] Test inference runs without errors
- [ ] Generated text output looks reasonable

---

## Troubleshooting

### Issue: "No module named 'transformers'"

**Solution**: Install transformers in MLX environment
```bash
source venv_mlx/bin/activate
pip install transformers tokenizers
```

### Issue: "Model weights don't match"

**Solution**: This is expected initially. The weight mapping in `convert_trocr_to_mlx.py` may need adjustment based on the actual PyTorch model structure. Check the console output for missing weights.

### Issue: "Out of memory during conversion"

**Solution**: The conversion process is memory-intensive. Close other applications or use a machine with more RAM.

### Issue: "Generated text is gibberish"

**Solution**: This indicates weight conversion issues. Common causes:
- Weight transposition errors (linear layers)
- Incorrect layer mapping
- Missing bias terms

To debug: Compare a few weights manually between PyTorch and MLX models.

---

## Performance Comparison

Once conversion is complete, you can benchmark:

**PyTorch (MPS)**:
```bash
# Run with original PyTorch model
python3 ocr_inference_handler.py
```

**MLX**:
```bash
# Run with MLX model
python3 scripts/mlx_inference.py --model models/mlx/trocr-base-handwritten --image test.jpg
```

**Expected improvements**:
- Inference speed: 2-3x faster
- Memory usage: ~30-40% reduction
- Model loading: 3-5x faster

---

## Important Notes

### Weight Mapping Complexity

The conversion script includes detailed weight mapping from PyTorch to MLX. Key differences:

1. **Linear Layer Weights**: PyTorch uses (out, in), MLX uses (in, out)
2. **Conv2D Weights**: Different channel ordering
3. **Embedding Layers**: Generally compatible, no transposition needed
4. **Layer Names**: PyTorch uses nested modules, MLX uses flat structure

### Current Limitations

1. **Beam Search**: Not implemented yet (greedy decoding only)
2. **Batch Processing**: Single image at a time
3. **Cross-Attention**: Simplified implementation
4. **Cache**: No KV-cache for faster generation

These can be added in future iterations.

---

## Next Steps

After successful conversion:

1. **Integrate with existing pipeline**:
   - Replace PyTorch inference in `ocr_inference_handler.py`
   - Update `ocr_processor_handler.py` to use MLX

2. **Optimize inference**:
   - Implement batch processing
   - Add KV-cache for generation
   - Optimize image preprocessing

3. **Training (Phase 4)**:
   - Implement MLX training loop
   - Convert fine-tuning pipeline

---

## Files Reference

### Created in Phase 2
```
scripts/
├── inspect_trocr_model.py       # Model inspection
├── mlx_models.py                 # MLX model architecture
├── convert_trocr_to_mlx.py      # Conversion script
└── mlx_inference.py             # MLX inference

models/
├── pytorch/
│   └── trocr-base-handwritten/  # Original PyTorch model
└── mlx/
    └── trocr-base-handwritten/  # Converted MLX model
```

### To Be Created in Phase 3
```
scripts/
├── mlx_ocr_processor.py         # MLX-based processor
└── mlx_batch_inference.py       # Batch processing
```

---

## Support

If you encounter issues during conversion:

1. Check the console output for specific errors
2. Verify all dependencies are installed in `venv_mlx`
3. Ensure sufficient disk space (~2GB for models)
4. Review the weight mapping in `convert_trocr_to_mlx.py`

For persistent issues, compare a small subset of weights manually between PyTorch and MLX models to identify mapping problems.

---

**Status**: Phase 2 scripts created ✅
**Next**: Run conversion steps above
**Then**: Phase 3 - Integration with existing pipeline
