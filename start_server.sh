#!/bin/bash

# GoScriptureAPI Startup Script
# This script handles ONNX Runtime library path and provides better error messages

set -e

# Show help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "GoScriptureAPI Startup Script"
    echo ""
    echo "Usage: $0 [PORT] [-debug]"
    echo ""
    echo "Arguments:"
    echo "  PORT     Port to listen on (default: 8080)"
    echo "  -debug   Enable debug logging"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start on port 8080"
    echo "  $0 3000              # Start on port 3000"
    echo "  $0 8080 -debug       # Start on port 8080 with debug logging"
    echo ""
    exit 0
fi

PORT=${1:-8080}
DEBUG_FLAG=""

# Validate port is numeric
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Port must be a number"
    echo "Usage: $0 [PORT] [-debug]"
    exit 1
fi

if [[ "$2" == "-debug" || "$*" == *"-debug"* ]]; then
    DEBUG_FLAG="-debug"
fi

echo "🚀 Starting GoScriptureAPI on port $PORT"
echo "📁 Working directory: $(pwd)"

# Check if binary exists
if [[ ! -f "./goscriptureapi" ]]; then
    echo "❌ Binary not found. Building..."
    go build -o goscriptureapi .
    echo "✅ Build complete"
fi

# Check for ONNX Runtime libraries
if [[ -f "./libonnxruntime.so" ]]; then
    echo "📚 Using local ONNX Runtime libraries"
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
elif ldconfig -p | grep -q onnxruntime; then
    echo "📚 Using system ONNX Runtime"
else
    echo "⚠️  ONNX Runtime not found!"
    echo "   This will fall back to simple embeddings (reduced quality)"
    echo "   See README.md for installation instructions"
fi

# Create symlinks if needed
if [[ -f "data/models/model.onnx" && ! -L "model.onnx" ]]; then
    echo "🔗 Creating model symlinks"
    ln -sf data/models/model.onnx .
    ln -sf data/models/model.onnx_data .
fi

echo "🎯 Starting server..."
echo "   Press Ctrl+C to stop"
echo "   API will be available at http://localhost:$PORT"
echo ""

./goscriptureapi -port $PORT $DEBUG_FLAG