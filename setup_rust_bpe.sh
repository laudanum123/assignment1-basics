#!/bin/bash
# Quick setup script for Rust BPE implementation

set -e

echo "=========================================="
echo "Rust BPE Implementation - Quick Setup"
echo "=========================================="

# Check for Rust
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "✓ Rust is installed: $(rustc --version)"
fi

# Check for maturin
echo ""
echo "Installing maturin..."
uv pip install maturin

# Build the Rust extension
echo ""
echo "Building Rust extension..."
maturin build --release

# Install the module
echo ""
echo "Installing module..."
WHEEL_FILE=$(ls -t target/wheels/*.whl | head -1)
python3 -m zipfile -e "$WHEEL_FILE" /tmp/bpe_wheel
cp -r /tmp/bpe_wheel/cs336_bpe_rust .venv/lib/python3.12/site-packages/

echo ""
echo "✅ Installation complete!"
echo ""
echo "Test with:"
echo "  source .venv/bin/activate"
echo "  python3 -c 'from cs336_bpe_rust import train_bpe_rust; print(\"Success!\")'"
echo ""
echo "Run benchmark:"
echo "  python3 benchmark_rust.py"
