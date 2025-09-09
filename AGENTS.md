# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Running
```bash
# Build the binary
go build -o goscriptureapi .

# Run with startup script (handles ONNX Runtime dependencies automatically)
./start_server.sh 8080 -debug

# Manual run with ONNX Runtime libraries
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./goscriptureapi -port 8080 -debug

# Build and run in one command
go run main.go -port 8080 -debug -data ./data
```

### Testing
```bash
# Run API integration tests
./test_api.sh

# Test specific endpoints
curl "http://localhost:8080/health"
curl "http://localhost:8080/status" 
curl "http://localhost:8080/search?q=love&k=3"

# Test with different ports (update test_api.sh ports as needed)
sed -i 's/8080/8083/g' test_api.sh  # Change test port
sed -i 's/8081/8083/g' test_api.sh  # Change test port
```

### Dependencies
```bash
# Install/update Go modules
go mod download
go mod tidy

# Install ONNX Runtime (required for EmbeddingGemma model)
# Fedora/RHEL: sudo dnf install onnxruntime onnxruntime-devel
# Ubuntu: sudo apt install libonnxruntime-dev
# Manual: Download from GitHub releases and copy to project directory
```

## Architecture Overview

### Core System Design
This is a **hybrid semantic search API** with two embedding backends:

1. **Primary**: EmbeddingGemma-300m ONNX model (real-time inference)
2. **Fallback**: SimpleEmbedding with pre-computed embeddings from Arweave

The architecture automatically attempts ONNX initialization in background and gracefully falls back to pre-computed embeddings if ONNX Runtime fails.

### Service Layer Hierarchy
```
main.go
├── EmbeddingService (embeddings/)
│   ├── RealONNXEmbeddingService (primary: EmbeddingGemma-300m)
│   └── SimpleEmbeddingService (fallback: pre-computed)
├── SearchService (search/)
│   ├── Vector indices (in-memory, float32/int8 quantized)
│   └── Granularity support (verse/chapter level)
└── API Handler (api/)
    └── Echo HTTP endpoints
```

### Critical ONNX Integration Points

**Model Files** (auto-downloaded to `data/models/`):
- `model.onnx` (480KB) - Architecture definition
- `model.onnx_data` (1.2GB) - EmbeddingGemma weights  
- `tokenizer.model` (4.6MB) - SentencePiece tokenizer

**ONNX Runtime Requirements**:
- Must have `libonnxruntime.so` accessible (system or local)
- Model files need symlinks in working directory: `model.onnx` → `data/models/model.onnx`
- Uses `DynamicSession[int64, float32]` for int64 token inputs, float32 embedding outputs
- Tensor shapes: input `[1, 512]` (batch_size, seq_length), output `[1, 768]` truncated to 128D

### Embedding Service Logic
The `EmbeddingService` in `embeddings/embeddings.go` coordinates between:

1. **RealONNXEmbeddingService** (`real_onnx_embeddings.go`):
   - SentencePiece tokenization with proper prefixes
   - ONNX model inference via `yalue/onnxruntime_go` 
   - Matryoshka truncation from 768D to 128D

2. **SimpleEmbeddingService** (`simple_embeddings.go`):
   - Pre-computed embeddings downloaded from Arweave URLs
   - Hash-based fallback embeddings for unknown text

**Embedding Flow**: `EmbedQuery/EmbedDocument` → try ONNX → fallback to Simple → fallback to hash-based

### Search Architecture
`SearchService` handles vector similarity search with:
- **Granularity**: verse-level (31K vectors) vs chapter-level (1.2K vectors)
- **Index Types**: float32 standard + int8 quantized for memory efficiency
- **Filtering**: book/chapter/verse filters applied post-search
- **Caching**: In-memory indices, on-demand loading from Arweave

### Configuration System
`config/config.go` provides:
- `ArweaveURLs`: Static URLs for pre-computed embeddings and text data
- `ModelConfig`: EmbeddingGemma model parameters (128D Matryoshka, query/document prefixes)
- Runtime config: port, data directory, debug mode

## Key Implementation Details

### ONNX Troubleshooting
- **Library Path Issues**: Use `LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH` 
- **Model File Issues**: Ensure `model.onnx_data` symlink exists in working directory
- **Initialization Failures**: Check logs for "Real ONNX EmbeddingGemma model initialized successfully"
- **Fallback Detection**: Watch for "Real ONNX service failed, falling back" debug messages

### Performance Characteristics  
- **ONNX Mode**: ~500ms query (400-600ms embedding + <1ms search), 1.4GB RAM
- **Fallback Mode**: ~100ms query, 20MB RAM
- **Memory**: Verse indices ~16MB unquantized, ~4MB int8 quantized

### Data Flow Patterns
1. **Startup**: Background ONNX init + preload verse/chapter indices from Arweave
2. **Query**: Parse filters → generate embedding (ONNX/fallback) → vector search → filter results
3. **Embedding**: Add task prefix → tokenize (SentencePiece) → ONNX inference → truncate to 128D

### Testing Strategy
The `test_api.sh` script tests both GET and POST search endpoints with various filter combinations. Note: script hardcodes ports 8080/8081 - update for different test ports.

### Startup Script Features
`start_server.sh` provides:
- Automatic ONNX Runtime detection (system vs local libraries)
- Binary building if needed
- Model symlink creation
- Library path configuration
- Helpful error messages for missing dependencies