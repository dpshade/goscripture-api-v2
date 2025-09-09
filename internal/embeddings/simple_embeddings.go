package embeddings

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/dpshade/goscriptureapi/internal/config"
	"github.com/rs/zerolog/log"
)

// SimpleEmbeddingService uses pre-computed embeddings with better query handling
type SimpleEmbeddingService struct {
	config         *config.Config
	verseEmbeddings map[string][]float32
	verseTexts     map[string]string
	initialized    bool
}

// NewSimpleEmbeddingService creates a better embedding service
func NewSimpleEmbeddingService(cfg *config.Config) (*SimpleEmbeddingService, error) {
	return &SimpleEmbeddingService{
		config:         cfg,
		verseEmbeddings: make(map[string][]float32),
		verseTexts:     make(map[string]string),
	}, nil
}

// Initialize loads the pre-computed embeddings
func (s *SimpleEmbeddingService) Initialize(verseEmbeddings map[string][]float32, verseTexts map[string]string) {
	s.verseEmbeddings = verseEmbeddings
	s.verseTexts = verseTexts
	s.initialized = true
	log.Info().Int("embeddings", len(verseEmbeddings)).Msg("Simple embedding service initialized")
}

// EmbedQuery generates embeddings for a search query by finding similar verses
func (s *SimpleEmbeddingService) EmbedQuery(text string) ([]float32, error) {
	if !s.initialized {
		return nil, fmt.Errorf("service not initialized")
	}

	// Find the most similar verses using text similarity
	bestMatches := s.findBestTextMatches(text, 5)
	
	if len(bestMatches) == 0 {
		return nil, fmt.Errorf("no similar verses found")
	}

	// Average the embeddings of the best matching verses
	return s.averageEmbeddings(bestMatches), nil
}

// EmbedDocument is not needed for our use case
func (s *SimpleEmbeddingService) EmbedDocument(text string) ([]float32, error) {
	return s.EmbedQuery(text) // Same as query for our purposes
}

// findBestTextMatches finds verses with similar text content
func (s *SimpleEmbeddingService) findBestTextMatches(query string, topK int) []string {
	type match struct {
		id    string
		score float64
	}

	queryWords := strings.Fields(strings.ToLower(query))
	var matches []match

	for verseID, verseText := range s.verseTexts {
		score := s.calculateTextSimilarity(queryWords, strings.ToLower(verseText))
		if score > 0 {
			matches = append(matches, match{id: verseID, score: score})
		}
	}

	// Sort by similarity score
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].score > matches[j].score
	})

	// Return top K matches
	var result []string
	for i, match := range matches {
		if i >= topK {
			break
		}
		result = append(result, match.id)
	}

	return result
}

// calculateTextSimilarity calculates similarity between query words and verse text
func (s *SimpleEmbeddingService) calculateTextSimilarity(queryWords []string, verseText string) float64 {
	verseWords := strings.Fields(verseText)
	verseWordSet := make(map[string]bool)
	
	for _, word := range verseWords {
		verseWordSet[word] = true
	}

	matchCount := 0
	for _, queryWord := range queryWords {
		if verseWordSet[queryWord] {
			matchCount++
		} else {
			// Check for partial matches
			for verseWord := range verseWordSet {
				if strings.Contains(verseWord, queryWord) || strings.Contains(queryWord, verseWord) {
					matchCount++
					break
				}
			}
		}
	}

	if len(queryWords) == 0 {
		return 0
	}

	// Calculate similarity as ratio of matching words with length bonus
	similarity := float64(matchCount) / float64(len(queryWords))
	
	// Bonus for longer verses (more context)
	lengthBonus := math.Log(float64(len(verseWords)+1)) / 10.0
	
	return similarity + lengthBonus
}

// averageEmbeddings computes the average of multiple embeddings
func (s *SimpleEmbeddingService) averageEmbeddings(verseIDs []string) []float32 {
	if len(verseIDs) == 0 {
		return make([]float32, config.ModelConfig.Dimensions)
	}

	// Initialize result with zeros
	result := make([]float32, config.ModelConfig.Dimensions)
	count := 0

	for _, verseID := range verseIDs {
		if embedding, exists := s.verseEmbeddings[verseID]; exists {
			for i, val := range embedding {
				if i < len(result) {
					result[i] += val
				}
			}
			count++
		}
	}

	// Average the embeddings
	if count > 0 {
		for i := range result {
			result[i] /= float32(count)
		}
	}

	return result
}