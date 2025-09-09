package embeddings

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/dpshade/goscriptureapi/internal/config"
	"github.com/eliben/go-sentencepiece"
	"github.com/rs/zerolog/log"
	ort "github.com/yalue/onnxruntime_go"
)

// RealONNXEmbeddingService implements EmbeddingGemma using proper ONNX Runtime and SentencePiece
type RealONNXEmbeddingService struct {
	config     *config.Config
	session    *ort.DynamicSession[int64, float32]
	tokenizer  *sentencepiece.Processor
	modelPath  string
	tokenizerPath string
	initialized bool
	mu         sync.RWMutex
}

// NewRealONNXEmbeddingService creates a new ONNX-based embedding service
func NewRealONNXEmbeddingService(cfg *config.Config) (*RealONNXEmbeddingService, error) {
	service := &RealONNXEmbeddingService{
		config: cfg,
	}

	return service, nil
}

// Initialize downloads and loads the model and tokenizer
func (s *RealONNXEmbeddingService) Initialize() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.initialized {
		return nil
	}

	// Create models directory
	modelDir := filepath.Join(s.config.DataDir, "models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	// Set up file paths
	s.modelPath = filepath.Join(modelDir, "model.onnx")
	s.tokenizerPath = filepath.Join(modelDir, "tokenizer.model")

	// Download model files if needed
	if err := s.downloadModelFiles(); err != nil {
		return fmt.Errorf("failed to download model files: %w", err)
	}

	// Load the ONNX model
	if err := s.loadONNXModel(); err != nil {
		return fmt.Errorf("failed to load ONNX model: %w", err)
	}

	// Load the SentencePiece tokenizer
	if err := s.loadTokenizer(); err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}

	s.initialized = true
	log.Info().Msg("Real ONNX EmbeddingGemma model initialized successfully")
	return nil
}

// downloadModelFiles downloads the ONNX model and tokenizer
func (s *RealONNXEmbeddingService) downloadModelFiles() error {
	files := map[string]string{
		s.modelPath: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx",
		s.modelPath + "_data": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx_data", // Model weights
		s.tokenizerPath: "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.model", // Use EmbeddingGemma's own tokenizer
	}

	for filePath, url := range files {
		if _, err := os.Stat(filePath); err == nil {
			log.Info().Str("path", filePath).Msg("File already exists")
			continue
		}

		log.Info().Str("url", url).Str("path", filePath).Msg("Downloading file...")

		if err := s.downloadFile(url, filePath); err != nil {
			return fmt.Errorf("failed to download %s: %w", filepath.Base(filePath), err)
		}

		log.Info().Str("path", filePath).Msg("File downloaded successfully")
	}

	return nil
}

// downloadFile downloads a file from URL to local path
func (s *RealONNXEmbeddingService) downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	return err
}

// loadONNXModel loads the ONNX model using ONNX Runtime
func (s *RealONNXEmbeddingService) loadONNXModel() error {
	// Initialize ONNX Runtime environment
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create session options
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	// Enable CPU optimizations
	if err := sessionOptions.SetIntraOpNumThreads(4); err != nil {
		log.Warn().Err(err).Msg("Failed to set intra-op threads")
	}

	// Create dynamic session for int64 input and float32 output data
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"sentence_embedding"}
	
	session, err := ort.NewDynamicSession[int64, float32](s.modelPath, inputNames, outputNames)
	if err != nil {
		return fmt.Errorf("failed to create ONNX session: %w", err)
	}

	s.session = session
	log.Info().Msg("ONNX model loaded successfully")
	return nil
}

// loadTokenizer loads the SentencePiece tokenizer
func (s *RealONNXEmbeddingService) loadTokenizer() error {
	// Read the tokenizer model file
	file, err := os.Open(s.tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to open tokenizer file: %w", err)
	}
	defer file.Close()

	tokenizer, err := sentencepiece.NewProcessor(file)
	if err != nil {
		return fmt.Errorf("failed to load SentencePiece tokenizer: %w", err)
	}

	s.tokenizer = tokenizer
	log.Info().Msg("SentencePiece tokenizer loaded successfully")
	return nil
}

// EmbedQuery generates embeddings for a search query
func (s *RealONNXEmbeddingService) EmbedQuery(text string) ([]float32, error) {
	prefixedText := config.ModelConfig.QueryPrefix + text
	return s.embed(prefixedText)
}

// EmbedDocument generates embeddings for a document
func (s *RealONNXEmbeddingService) EmbedDocument(text string) ([]float32, error) {
	prefixedText := config.ModelConfig.DocumentPrefix + text
	return s.embed(prefixedText)
}

// embed generates embeddings using the ONNX model
func (s *RealONNXEmbeddingService) embed(text string) ([]float32, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.initialized {
		return nil, fmt.Errorf("model not initialized")
	}

	// Tokenize the text
	tokens := s.tokenizer.Encode(text)
	
	// Prepare input with padding/truncation
	maxLength := 512
	inputIds := make([]int64, maxLength)
	attentionMask := make([]int64, maxLength)
	
	// Copy tokens (truncate if too long)
	copyLen := len(tokens)
	if copyLen > maxLength {
		copyLen = maxLength
	}
	
	for i := 0; i < copyLen; i++ {
		inputIds[i] = int64(tokens[i].ID)
		attentionMask[i] = 1
	}

	// Create input shapes
	batchSize := int64(1)
	seqLength := int64(maxLength)
	inputShape := []int64{batchSize, seqLength}
	
	// Create input tensors with int64 data type (as expected by the model)
	inputIdsTensor, err := ort.NewTensor(inputShape, inputIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	attentionTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer attentionTensor.Destroy()

	// Create output tensor (empty, will be populated by inference)
	// EmbeddingGemma outputs 768-dimensional embeddings
	outputShape := []int64{batchSize, 768}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run the ONNX model
	err = s.session.Run([]*ort.Tensor[int64]{inputIdsTensor, attentionTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("failed to run ONNX model: %w", err)
	}

	// Get the embedding data
	embeddingSlice := outputTensor.GetData()

	// The output should be [batch_size, hidden_size]. We only have batch_size=1
	if len(embeddingSlice) < config.ModelConfig.Dimensions {
		return nil, fmt.Errorf("output embedding size mismatch: got %d, expected %d", len(embeddingSlice), config.ModelConfig.Dimensions)
	}

	// Truncate to the desired dimensions (Matryoshka truncation to 128D)
	result := make([]float32, config.ModelConfig.Dimensions)
	copy(result, embeddingSlice[:config.ModelConfig.Dimensions])

	return result, nil
}

// Close cleans up resources
func (s *RealONNXEmbeddingService) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}

	// SentencePiece processor doesn't need explicit cleanup
	s.tokenizer = nil

	ort.DestroyEnvironment()
	s.initialized = false

	return nil
}