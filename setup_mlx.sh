#!/bin/bash
# ============================================
# MLX OCR Setup Script
# ============================================
# Automated setup for Apple MLX-based OCR system
# Optimized for macOS with Apple Silicon (M4 Max)
# Author: Diego Alarcon
# Date: October 2025
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# Helper Functions
# ============================================
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  MLX OCR Setup - Apple Silicon Optimized                ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# ============================================
# System Checks
# ============================================
print_header

print_step "Checking system requirements..."

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only"
    exit 1
fi
print_success "Running on macOS"

# Check Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    print_warning "Not running on Apple Silicon - MLX performance will be limited"
else
    print_success "Apple Silicon detected (optimal performance)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"
if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    print_error "Python 3.9+ required. Current: $PYTHON_VERSION"
    exit 1
fi
print_success "Python $PYTHON_VERSION detected"

echo ""

# ============================================
# Virtual Environment Setup
# ============================================
print_step "Setting up virtual environment..."

VENV_NAME="venv_mlx"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        print_success "Removed existing environment"
    else
        print_step "Using existing environment"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created: $VENV_NAME"
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"
print_success "Virtual environment activated"

echo ""

# ============================================
# Upgrade pip
# ============================================
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
print_success "pip upgraded to latest version"

echo ""

# ============================================
# Install MLX Dependencies
# ============================================
print_step "Installing MLX and dependencies..."
echo ""

# Install in stages for better error handling
print_step "  [1/4] Installing MLX core..."
pip install mlx==0.22.0 --quiet
print_success "  MLX core installed"

print_step "  [2/4] Installing MLX-LM..."
pip install mlx-lm==0.21.1 --quiet
print_success "  MLX-LM installed"

print_step "  [3/4] Installing image processing libraries..."
pip install opencv-python==4.10.0.84 numpy==2.1.2 Pillow==10.4.0 matplotlib==3.9.2 --quiet
print_success "  Image processing libraries installed"

print_step "  [4/4] Installing utilities and tools..."
pip install -r requirements-mlx.txt --quiet
print_success "  All dependencies installed"

echo ""

# ============================================
# Verify Installation
# ============================================
print_step "Verifying MLX installation..."

python3 << EOF
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import cv2
from PIL import Image

# Test MLX
a = mx.array([1, 2, 3])
print(f"MLX version: {mx.__version__}")
print(f"MLX test array: {a}")

# Test image libs
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PIL version: {Image.__version__}")

print("\n✓ All core libraries working correctly")
EOF

if [ $? -eq 0 ]; then
    print_success "MLX installation verified"
else
    print_error "MLX verification failed"
    exit 1
fi

echo ""

# ============================================
# Create Directory Structure
# ============================================
print_step "Creating directory structure..."

mkdir -p models/mlx
mkdir -p models/converted
mkdir -p output/mlx
mkdir -p temp

print_success "Directory structure created"

echo ""

# ============================================
# Summary
# ============================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  Setup Complete!                                          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Activate environment: ${YELLOW}source $VENV_NAME/bin/activate${NC}"
echo "  2. Convert your TrOCR model to MLX format"
echo "  3. Run MLX-based inference"
echo ""
echo -e "${GREEN}Installed Packages:${NC}"
pip list | grep -E "(mlx|opencv|numpy|pillow|transformers)" | sed 's/^/  /'
echo ""
echo -e "${GREEN}Environment Info:${NC}"
echo "  Virtual Environment: $VENV_NAME"
echo "  Python: $(which python3)"
echo "  Platform: $(uname -m)"
echo ""
print_success "Ready for MLX OCR processing!"
