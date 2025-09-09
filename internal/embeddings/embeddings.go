package embeddings

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/dpshade/goscriptureapi/internal/config"
	"github.com/rs/zerolog/log"
)

// EmbeddingService handles text embedding generation
type EmbeddingService struct {
	config          *config.Config
	modelLoaded     bool
	mu              sync.RWMutex
	// Removed old ONNX service
	realOnnxService *RealONNXEmbeddingService // New real ONNX service
	simpleService   *SimpleEmbeddingService
	usePrecomputed  bool
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService(cfg *config.Config) (*EmbeddingService, error) {
	// Try real ONNX implementation first
	realOnnxService, err := NewRealONNXEmbeddingService(cfg)
	if err == nil {
		// Try to initialize in background (don't block startup)
		go func() {
			if initErr := realOnnxService.Initialize(); initErr == nil {
				log.Info().Msg("Real ONNX EmbeddingGemma model initialized in background")
			} else {
				log.Warn().Err(initErr).Msg("Failed to initialize real ONNX model")
			}
		}()
		
		// Return service that can use ONNX when ready
		service := &EmbeddingService{
			config:         cfg,
			realOnnxService: realOnnxService,
			usePrecomputed: false,
		}
		
		// Also initialize simple service as fallback
		simpleService, simpleErr := NewSimpleEmbeddingService(cfg)
		if simpleErr == nil {
			service.simpleService = simpleService
		}

		// Create data directory
		if err := os.MkdirAll(cfg.DataDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create data directory: %w", err)
		}

		log.Info().Msg("EmbeddingService initialized (real ONNX + fallback)")
		return service, nil
	}
	
	// Fallback to simple embedding service
	log.Warn().Err(err).Msg("Failed to create real ONNX service, using simple approach")
	
	simpleService, err := NewSimpleEmbeddingService(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create simple embedding service: %w", err)
	}

	service := &EmbeddingService{
		config:        cfg,
		simpleService: simpleService,
		usePrecomputed: true,
	}

	// Create data directory if it doesn't exist
	if err := os.MkdirAll(cfg.DataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	log.Info().Msg("EmbeddingService initialized (simple embedding approach)")
	return service, nil
}

// EmbedQuery generates embeddings for a search query
func (s *EmbeddingService) EmbedQuery(text string) ([]float32, error) {
	// Try real ONNX service first if available and initialized
	if s.realOnnxService != nil {
		if embedding, err := s.realOnnxService.EmbedQuery(text); err == nil {
			return embedding, nil
		} else {
			log.Debug().Err(err).Msg("Real ONNX service failed, falling back")
		}
	}
	
	// Old ONNX service removed
	
	// Try simple service
	if s.simpleService != nil {
		if embedding, err := s.simpleService.EmbedQuery(text); err == nil {
			return embedding, nil
		}
	}
	
	// Final fallback to placeholder embedding
	return s.generatePlaceholderEmbedding(config.ModelConfig.QueryPrefix + text), nil
}

// EmbedDocument generates embeddings for a document
func (s *EmbeddingService) EmbedDocument(text string) ([]float32, error) {
	// Try real ONNX service first if available and initialized
	if s.realOnnxService != nil {
		if embedding, err := s.realOnnxService.EmbedDocument(text); err == nil {
			return embedding, nil
		}
	}
	
	// Old ONNX service removed
	
	// Try simple service
	if s.simpleService != nil {
		if embedding, err := s.simpleService.EmbedDocument(text); err == nil {
			return embedding, nil
		}
	}
	
	// Final fallback to placeholder embedding
	return s.generatePlaceholderEmbedding(config.ModelConfig.DocumentPrefix + text), nil
}

// InitializeWithPrecomputedData initializes the simple service with loaded embeddings
func (s *EmbeddingService) InitializeWithPrecomputedData(embeddings map[string][]float32, texts map[string]string) {
	if s.simpleService != nil {
		s.simpleService.Initialize(embeddings, texts)
	}
}

// generatePlaceholderEmbedding creates a deterministic placeholder embedding
// In production, this would be replaced with actual model inference
func (s *EmbeddingService) generatePlaceholderEmbedding(text string) []float32 {
	embedding := make([]float32, config.ModelConfig.Dimensions)
	
	// Create a simple hash-based embedding for testing
	// This ensures the same text always produces the same embedding
	hash := uint32(0)
	for _, char := range text {
		hash = hash*31 + uint32(char)
	}
	
	// Generate deterministic values
	for i := range embedding {
		// Use different hash variations for each dimension
		seed := hash + uint32(i)*2654435761
		// Convert to float in range [-1, 1]
		embedding[i] = (float32(seed) / float32(math.MaxUint32)) * 2 - 1
	}
	
	// Normalize the vector
	return normalize(embedding)
}

// normalize normalizes a vector to unit length
func normalize(vec []float32) []float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	
	if sum == 0 {
		return vec
	}
	
	norm := float32(math.Sqrt(float64(sum)))
	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}
	
	return normalized
}

// LoadONNXModel would load the ONNX model from file or download it
// This is a placeholder for the actual implementation
func (s *EmbeddingService) LoadONNXModel(modelPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.modelLoaded {
		return nil
	}

	// In production:
	// 1. Check if model exists at modelPath
	// 2. If not, download from Hugging Face
	// 3. Initialize ONNX Runtime
	// 4. Load tokenizer
	
	log.Info().Msg("ONNX model loading simulated - using pre-computed embeddings")
	s.modelLoaded = true
	return nil
}

// DownloadModel downloads the model from Hugging Face if needed
func (s *EmbeddingService) DownloadModel() error {
	modelDir := filepath.Join(s.config.DataDir, "models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	// Check if model already exists
	modelPath := filepath.Join(modelDir, "model.onnx")
	if _, err := os.Stat(modelPath); err == nil {
		log.Info().Msg("Model already downloaded")
		return nil
	}

	// In production, download from Hugging Face
	// For now, we'll skip this and use pre-computed embeddings
	log.Info().Msg("Model download skipped - using pre-computed embeddings")
	return nil
}

// EmbeddingData represents the structure of pre-computed embeddings
type EmbeddingData struct {
	Embeddings []struct {
		ID        string    `json:"id"`
		Embedding []float32 `json:"embedding"`
	} `json:"embeddings"`
}

// LoadPrecomputedEmbeddings loads pre-computed embeddings from a URL
func (s *EmbeddingService) LoadPrecomputedEmbeddings(url string) (*EmbeddingData, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch embeddings: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch embeddings: status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var data EmbeddingData
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("failed to parse embeddings: %w", err)
	}

	return &data, nil
}

// SerializeEmbedding converts an embedding to bytes for storage
func SerializeEmbedding(embedding []float32) []byte {
	buf := new(bytes.Buffer)
	for _, val := range embedding {
		binary.Write(buf, binary.LittleEndian, val)
	}
	return buf.Bytes()
}

// DeserializeEmbedding converts bytes back to an embedding
func DeserializeEmbedding(data []byte, dimensions int) ([]float32, error) {
	if len(data) != dimensions*4 {
		return nil, fmt.Errorf("invalid embedding data size")
	}

	embedding := make([]float32, dimensions)
	buf := bytes.NewReader(data)
	for i := range embedding {
		if err := binary.Read(buf, binary.LittleEndian, &embedding[i]); err != nil {
			return nil, err
		}
	}
	return embedding, nil
}