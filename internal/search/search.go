package search

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/dpshade/goscriptureapi/internal/config"
	"github.com/dpshade/goscriptureapi/internal/embeddings"
	"github.com/rs/zerolog/log"
)

// SearchService handles semantic search operations
type SearchService struct {
	embeddings      *embeddings.EmbeddingService
	config          *config.Config
	indices         map[string]*VectorIndex
	textLookup      map[string]map[string]*TextData
	loadedGranularities map[string]bool
	mu              sync.RWMutex
	cache           *Cache
}

// TextData represents the text and metadata for a verse or chapter
type TextData struct {
	Text string   `json:"text"`
	Meta Metadata `json:"meta"`
}

// Metadata contains metadata for a text chunk
type Metadata struct {
	Reference string   `json:"reference,omitempty"`
	Book      string   `json:"book"`
	Chapter   int      `json:"chapter"`
	VerseNum  int      `json:"verseNum,omitempty"`
	Events    []string `json:"events,omitempty"`
	Entities  []string `json:"entities,omitempty"`
}

// SearchResult represents a search result
type SearchResult struct {
	ID         string    `json:"id"`
	Similarity float32   `json:"similarity"`
	Score      float32   `json:"score"`
	Chunk      ChunkData `json:"chunk"`
}

// ChunkData represents the data for a search result chunk
type ChunkData struct {
	ID   string   `json:"id"`
	Text string   `json:"text"`
	Meta Metadata `json:"meta"`
}

// SearchOptions contains options for search
type SearchOptions struct {
	Book        string `json:"book,omitempty"`
	Chapter     string `json:"chapter,omitempty"`
	Verse       string `json:"verse,omitempty"`
	Granularity string `json:"granularity,omitempty"` // "verse" or "chapter"
	K           int    `json:"k,omitempty"`           // Number of results
}

// Cache provides simple in-memory caching
type Cache struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

// NewCache creates a new cache
func NewCache() *Cache {
	return &Cache{
		data: make(map[string]interface{}),
	}
}

// Get retrieves a value from cache
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	val, ok := c.data[key]
	return val, ok
}

// Set stores a value in cache
func (c *Cache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

// NewSearchService creates a new search service
func NewSearchService(embeddingService *embeddings.EmbeddingService, cfg *config.Config) (*SearchService, error) {
	service := &SearchService{
		embeddings:          embeddingService,
		config:             cfg,
		indices:            make(map[string]*VectorIndex),
		textLookup:         make(map[string]map[string]*TextData),
		loadedGranularities: make(map[string]bool),
		cache:              NewCache(),
	}

	return service, nil
}

// PreloadGranularity loads embeddings and text data for a granularity
func (s *SearchService) PreloadGranularity(granularity string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.loadedGranularities[granularity] {
		log.Info().Str("granularity", granularity).Msg("Granularity already loaded")
		return nil
	}

	var embeddingURL, fallbackURL, textURL string
	
	switch granularity {
	case "verse":
		embeddingURL = config.ArweaveURLs.Verses
		fallbackURL = config.ArweaveURLs.VersesUncompressed
		textURL = config.ArweaveURLs.VerseText
	case "chapter":
		embeddingURL = config.ArweaveURLs.Chapters
		fallbackURL = config.ArweaveURLs.ChaptersUncompressed
		textURL = config.ArweaveURLs.ChapterText
	default:
		return fmt.Errorf("unknown granularity: %s", granularity)
	}

	// Try to load from cache first
	cacheDir := filepath.Join(s.config.DataDir, "cache", granularity)
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		log.Warn().Err(err).Msg("Failed to create cache directory")
	}

	// Load embeddings
	embeddingData, err := s.loadWithFallback(embeddingURL, fallbackURL)
	if err != nil {
		return fmt.Errorf("failed to load embeddings: %w", err)
	}

	// Load text data
	textData, err := s.loadFromURL(textURL, false)
	if err != nil {
		return fmt.Errorf("failed to load text data: %w", err)
	}

	// Parse and store embeddings
	index := NewVectorIndex()
	embeddings := make(map[string][]float32)
	texts := make(map[string]string)
	
	if embeddingData, ok := embeddingData.(map[string]interface{}); ok {
		if embeddingsList, ok := embeddingData["embeddings"].([]interface{}); ok {
			for _, item := range embeddingsList {
				if embedding, ok := item.(map[string]interface{}); ok {
					id := embedding["id"].(string)
					if vecData, ok := embedding["embedding"].([]interface{}); ok {
						vec := make([]float32, len(vecData))
						for i, v := range vecData {
							if f, ok := v.(float64); ok {
								vec[i] = float32(f)
							}
						}
						index.Add(id, vec)
						embeddings[id] = vec
					}
				}
			}
		}
	}

	s.indices[granularity] = index

	// Process text data
	textLookup := s.processTextData(textData, granularity)
	s.textLookup[granularity] = textLookup
	
	// Extract text strings for the embedding service
	for id, textData := range textLookup {
		texts[id] = textData.Text
	}
	
	// Initialize the embedding service with this data
	if granularity == "verse" {
		s.embeddings.InitializeWithPrecomputedData(embeddings, texts)
	}

	s.loadedGranularities[granularity] = true
	
	log.Info().
		Str("granularity", granularity).
		Int("vectors", index.Size()).
		Msg("Granularity loaded successfully")

	return nil
}

// loadWithFallback tries to load from primary URL, falls back to secondary if needed
func (s *SearchService) loadWithFallback(primaryURL, fallbackURL string) (interface{}, error) {
	// Try compressed version first
	data, err := s.loadFromURL(primaryURL, true)
	if err == nil {
		return data, nil
	}

	log.Warn().Err(err).Msg("Primary URL failed, trying fallback")
	
	// Try uncompressed fallback
	return s.loadFromURL(fallbackURL, false)
}

// loadFromURL loads data from a URL
func (s *SearchService) loadFromURL(url string, compressed bool) (interface{}, error) {
	// Check cache first
	if cached, ok := s.cache.Get(url); ok {
		return cached, nil
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var reader io.Reader = resp.Body
	
	// Handle gzip compression
	if compressed {
		// Check if content is gzipped
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		
		// Check gzip magic number
		if len(body) >= 2 && body[0] == 0x1f && body[1] == 0x8b {
			gzReader, err := gzip.NewReader(strings.NewReader(string(body)))
			if err != nil {
				return nil, err
			}
			defer gzReader.Close()
			reader = gzReader
		} else {
			reader = strings.NewReader(string(body))
		}
	}

	// Read and parse JSON
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	var result interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Cache the result
	s.cache.Set(url, result)

	return result, nil
}

// processTextData processes text data into a lookup structure
func (s *SearchService) processTextData(data interface{}, granularity string) map[string]*TextData {
	lookup := make(map[string]*TextData)

	// Handle array of verses
	if verses, ok := data.([]interface{}); ok {
		for i, item := range verses {
			if verse, ok := item.(map[string]interface{}); ok {
				textData := &TextData{
					Text: getStringField(verse, "text"),
					Meta: Metadata{
						Reference: getStringField(verse, "ref"),
						Book:      getStringField(verse, "book"),
						Chapter:   getIntField(verse, "chapter"),
						VerseNum:  getIntField(verse, "verseNum"),
					},
				}

				// Add events and entities if present
				if events, ok := verse["events"].([]interface{}); ok {
					textData.Meta.Events = interfaceSliceToStringSlice(events)
				}
				if entities, ok := verse["entities"].([]interface{}); ok {
					textData.Meta.Entities = interfaceSliceToStringSlice(entities)
				}

				// Store under multiple possible ID formats
				ref := textData.Meta.Reference
				book := textData.Meta.Book
				chapter := textData.Meta.Chapter
				verseNum := textData.Meta.VerseNum

				possibleIDs := []string{
					ref,
					fmt.Sprintf("verse:%s:%d:%d", book, chapter, verseNum),
					fmt.Sprintf("chapter:%s:%d", book, chapter),
					fmt.Sprintf("%s_%d", granularity, i),
					fmt.Sprintf("%s.%d.%d", book, chapter, verseNum),
					fmt.Sprintf("%s_%d_%d", book, chapter, verseNum),
					fmt.Sprintf("%d", i),
					fmt.Sprintf("v%d", i),
				}

				for _, id := range possibleIDs {
					if id != "" {
						lookup[id] = textData
					}
				}
			}
		}
	}

	return lookup
}

// Search performs semantic search
func (s *SearchService) Search(query string, options SearchOptions) ([]SearchResult, error) {
	if query == "" {
		return nil, nil
	}

	// Default options
	if options.Granularity == "" {
		options.Granularity = "verse"
	}
	if options.K == 0 {
		options.K = 10
	}

	// Check if granularity is loaded
	s.mu.RLock()
	if !s.loadedGranularities[options.Granularity] {
		s.mu.RUnlock()
		return nil, fmt.Errorf("granularity %s not loaded", options.Granularity)
	}
	index := s.indices[options.Granularity]
	textLookup := s.textLookup[options.Granularity]
	s.mu.RUnlock()

	// Generate query embedding using the real model
	queryEmbedding, err := s.embeddings.EmbedQuery(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Create filter function if filters are specified
	var filterFunc func(id string) bool
	if options.Book != "" || options.Chapter != "" {
		filterFunc = func(id string) bool {
			if text, ok := textLookup[id]; ok {
				if options.Book != "" && !strings.EqualFold(text.Meta.Book, options.Book) {
					return false
				}
				if options.Chapter != "" && fmt.Sprintf("%d", text.Meta.Chapter) != options.Chapter {
					return false
				}
				return true
			}
			return false
		}
	} else {
		// No filter - accept all
		filterFunc = func(id string) bool { return true }
	}

	// Search the index
	searchResults := index.SearchWithFilter(queryEmbedding, options.K, filterFunc)

	// Convert to final results with text
	results := make([]SearchResult, 0, len(searchResults))
	for _, sr := range searchResults {
		textData, ok := textLookup[sr.ID]
		if !ok {
			// Create placeholder if text not found
			textData = &TextData{
				Text: fmt.Sprintf("[Text not found for ID: %s]", sr.ID),
				Meta: Metadata{},
			}
		}

		results = append(results, SearchResult{
			ID:         sr.ID,
			Similarity: sr.Similarity,
			Score:      sr.Score,
			Chunk: ChunkData{
				ID:   sr.ID,
				Text: textData.Text,
				Meta: textData.Meta,
			},
		})
	}

	return results, nil
}

// GetStatus returns the current status of the search service
func (s *SearchService) GetStatus() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()

	status := map[string]interface{}{
		"initialized": true,
		"indices": make(map[string]interface{}),
	}

	for granularity, index := range s.indices {
		status["indices"].(map[string]interface{})[granularity] = map[string]interface{}{
			"loaded": s.loadedGranularities[granularity],
			"count":  index.Size(),
			"memoryBytes": index.GetMemoryUsage(),
		}
	}

	return status
}

// Helper functions
func getStringField(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getIntField(m map[string]interface{}, key string) int {
	if val, ok := m[key].(float64); ok {
		return int(val)
	}
	if val, ok := m[key].(int); ok {
		return val
	}
	if val, ok := m[key].(string); ok {
		// Try to parse string as int
		var intVal int
		fmt.Sscanf(val, "%d", &intVal)
		return intVal
	}
	return 0
}

func interfaceSliceToStringSlice(slice []interface{}) []string {
	result := make([]string, 0, len(slice))
	for _, item := range slice {
		if str, ok := item.(string); ok {
			result = append(result, str)
		}
	}
	return result
}