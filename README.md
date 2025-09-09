# GoScriptureAPI

A high-performance Go HTTP API for semantic Bible search, mimicking the functionality of the scripture frontend JS application with persistent in-memory indices for optimal performance.

## Features

- **Semantic Search**: Uses EmbeddingGemma-300M model at 128 dimensions (Matryoshka truncation)
- **Pre-computed Embeddings**: Leverages pre-computed embeddings from Arweave for fast initialization
- **In-Memory Indices**: Keeps vector indices in memory for sub-millisecond search latency
- **Quantization Support**: Memory-efficient storage using int8 quantization
- **Granularity Options**: Search at verse or chapter level
- **Filtering**: Support for book, chapter, and verse filters
- **RESTful API**: Echo framework for high-performance HTTP handling

## Architecture

The API is designed to mirror the semantic search capabilities of the JavaScript frontend while providing:
- Persistent in-memory storage (indices stay loaded between requests)
- Better resource utilization through Go's efficient memory management
- Support for advanced quantization techniques
- Horizontal scalability potential

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API.

### Status
```
GET /status
```
Returns detailed status information including loaded indices and memory usage.

### Search

**GET Request** (Recommended):
```
GET /search?q=God%20loved%20the%20world&k=10&book=John&chapter=3
```

**POST Request** (Also supported):
```
POST /search
Content-Type: application/json

{
  "query": "God loved the world",
  "options": {
    "book": "John",
    "chapter": "3",
    "granularity": "verse",
    "k": 10
  }
}
```

**GET Query Parameters:**
- `q` or `query` - Search query text (required)
- `k` - Number of results (default: 10)
- `book` - Filter by Bible book
- `chapter` - Filter by chapter number
- `verse` - Filter by verse number  
- `granularity` - Search granularity: "verse" or "chapter" (default: "verse")

Alternatively, filters can be embedded in the query text:
```
GET /search?q=love%20your%20enemies%20book:Matthew%20chapter:5
```

Response:
```json
{
  "query": "God loved the world",
  "results": [
    {
      "book": "John",
      "chapter": 3,
      "verseNum": 16,
      "text": "For God so loved the world...",
      "_searchMeta": {
        "similarity": 0.95,
        "score": 0.95,
        "reference": "John 3:16"
      }
    }
  ],
  "count": 10,
  "status": "success"
}
```

### Embed (Planned)
```
POST /embed
Content-Type: application/json

{
  "text": "sample text",
  "type": "query"
}
```
This endpoint will generate embeddings directly once ONNX runtime is integrated.

## Quick Start

### Prerequisites

**ONNX Runtime Installation (Required for EmbeddingGemma)**

The server now uses the real EmbeddingGemma-300m ONNX model instead of pre-computed embeddings. ONNX Runtime is required:

**Option 1: System Package (Recommended)**
```bash
# Fedora/RHEL/CentOS
sudo dnf install onnxruntime onnxruntime-devel

# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# macOS
brew install onnxruntime
```

**Option 2: Manual Installation**
```bash
# Download ONNX Runtime (Linux x64 example)
curl -L -O "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz"
tar -xzf onnxruntime-linux-x64-1.19.2.tgz

# Copy libraries to system or local directory
sudo cp onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig

# Or copy to project directory (see Running section below)
```

### Running with Go

**Standard Installation (with system ONNX Runtime):**
```bash
# Install dependencies
go mod download

# Build the server
go build -o goscriptureapi

# Run the server
./goscriptureapi -port 8080 -debug
```

**Local ONNX Runtime Installation:**
```bash
# If ONNX Runtime is not system-installed, copy libraries to project directory
cp path/to/onnxruntime-linux-x64-1.19.2/lib/libonnxruntime.so* .
ln -sf libonnxruntime.so onnxruntime.so

# Run with library path
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./goscriptureapi -port 8080 -debug
```

**Easy Startup Script:**
```bash
# Use the included startup script for automatic configuration
./start_server.sh 8080          # Normal mode
./start_server.sh 8080 -debug   # Debug mode

# The script automatically:
# - Builds the binary if needed
# - Detects ONNX Runtime installation
# - Sets up library paths and symlinks
# - Provides helpful status messages
```

### First-Time Setup

The server will automatically download required model files on first run:

1. **model.onnx** (~480KB) - Model architecture
2. **model.onnx_data** (~1.2GB) - EmbeddingGemma-300m weights  
3. **tokenizer.model** (~4.6MB) - SentencePiece tokenizer

These files are cached in `data/models/` for subsequent runs.

**Expected startup logs:**
```
INFO Initializing EmbeddingGemma service...
INFO Downloading file... path=data/models/model.onnx_data url=https://huggingface.co/...
INFO File downloaded successfully path=data/models/model.onnx_data
INFO ONNX model loaded successfully
INFO SentencePiece tokenizer loaded successfully  
INFO Real ONNX EmbeddingGemma model initialized successfully
```

### Command Line Options
- `-port`: Port to listen on (default: 8080)
- `-data`: Directory to store cached data and models (default: ./data)
- `-debug`: Enable debug logging

### Troubleshooting

**"onnxruntime.so: cannot open shared object file"**
```bash
# Option 1: Install system-wide
sudo dnf install onnxruntime  # Fedora
sudo apt install libonnxruntime-dev  # Ubuntu

# Option 2: Use local libraries
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./goscriptureapi
```

**"model.onnx_data: No such file or directory"**
```bash
# Create symlinks in working directory
ln -sf data/models/model.onnx .
ln -sf data/models/model.onnx_data .
```

**Model download failures**
- Check internet connectivity to huggingface.co
- Ensure sufficient disk space (~1.3GB for model files)
- Verify write permissions in data/models directory

### Running with Docker
```bash
# Build the image
docker build -t goscriptureapi .

# Run the container
docker run -p 8080:8080 goscriptureapi
```

## Development

### Project Structure
```
.
├── main.go                 # Entry point
├── internal/
│   ├── api/               # HTTP handlers
│   ├── config/            # Configuration
│   ├── embeddings/        # Embedding generation service
│   └── search/            # Search service and vector index
├── data/                  # Cached data directory
└── go.mod                 # Go module definition
```

### Implementation Notes

1. **Embedding Model**: **Now uses real EmbeddingGemma-300m ONNX model** for generating embeddings on-the-fly. This provides superior semantic understanding compared to pre-computed embeddings.

2. **Vector Index**: Implements both standard float32 and quantized int8 indices for memory efficiency.

3. **Hybrid Architecture**: Falls back to SimpleEmbeddingService with pre-computed embeddings if ONNX model fails to initialize (for development/debugging).

4. **Caching**: 
   - Model files (1.3GB total) cached locally in `data/models/`
   - Pre-computed embeddings from Arweave for fallback mode
   - Generated embeddings are computed on-demand (not cached)

5. **Concurrency**: Indices and model initialization run in background goroutines for fast startup.

## Performance Considerations

### EmbeddingGemma ONNX Mode (Primary)
- **Memory Usage**: 
  - Model weights: ~1.2GB (loaded once)
  - ONNX Runtime overhead: ~200MB  
  - Verse indices: ~16MB unquantized, ~4MB with int8 quantization
  - Total: **~1.4GB RAM**

- **Search Latency**: 
  - Embedding generation: ~400-600ms (EmbeddingGemma inference)
  - Vector search: <1ms (in-memory operations)
  - **Total query time: ~500ms**

- **Initialization**: 
  - First-time: Downloads 1.3GB model files from Hugging Face
  - Subsequent: ~30 seconds for model loading
  - Pre-computed indices: Downloads ~20MB from Arweave

### Fallback Mode (SimpleEmbedding)
- **Memory Usage**: ~20MB total
- **Search Latency**: <100ms end-to-end  
- **Initialization**: Downloads ~20MB from Arweave only

### Hardware Requirements
- **Minimum**: 2GB RAM, 2GB disk space
- **Recommended**: 4GB+ RAM, SSD storage
- **CPU**: Any modern x86_64 processor (no GPU required)

## Future Enhancements

1. **~~ONNX Runtime Integration~~**: ✅ **Complete** - Full EmbeddingGemma-300m ONNX model integration
2. **GPU Acceleration**: Optional CUDA/ROCm support for faster embedding generation
3. **Advanced Quantization**: Product quantization and other techniques for even better memory efficiency
4. **Embedding Caching**: Cache generated embeddings to improve repeated query performance
5. **Distributed Index**: Support for sharding across multiple instances
6. **Batch Processing**: Batch multiple embeddings in single ONNX inference for better throughput
7. **Model Variants**: Support for different EmbeddingGemma sizes (768D full model)
8. **Incremental Updates**: Support for adding new texts without full reindexing

## Compatibility

This API is designed to be a drop-in replacement for the JavaScript semantic search, maintaining the same:
- Query syntax and filtering options
- Result format and scoring
- Embedding model and dimensions (EmbeddingGemma at 128D)

## License

MIT# goscripture-api-v2
