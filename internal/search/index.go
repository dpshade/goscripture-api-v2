package search

import (
	"encoding/json"
	"math"
	"sort"
	"sync"
)

// VectorIndex represents an in-memory vector index
type VectorIndex struct {
	Vectors [][]float32
	IDs     []string
	mu      sync.RWMutex
}

// NewVectorIndex creates a new vector index
func NewVectorIndex() *VectorIndex {
	return &VectorIndex{
		Vectors: make([][]float32, 0),
		IDs:     make([]string, 0),
	}
}

// Add adds a vector to the index
func (vi *VectorIndex) Add(id string, vector []float32) {
	vi.mu.Lock()
	defer vi.mu.Unlock()
	
	vi.IDs = append(vi.IDs, id)
	vi.Vectors = append(vi.Vectors, vector)
}

// Search performs a k-nearest neighbor search
func (vi *VectorIndex) Search(query []float32, k int) []SearchResult {
	vi.mu.RLock()
	defer vi.mu.RUnlock()
	
	if len(vi.Vectors) == 0 {
		return nil
	}
	
	// Calculate similarities for all vectors
	results := make([]SearchResult, 0, len(vi.Vectors))
	for i, vec := range vi.Vectors {
		similarity := cosineSimilarity(query, vec)
		results = append(results, SearchResult{
			ID:         vi.IDs[i],
			Similarity: similarity,
			Score:      similarity,
		})
	}
	
	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	
	// Return top k results
	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// SearchWithFilter performs a filtered k-nearest neighbor search
func (vi *VectorIndex) SearchWithFilter(query []float32, k int, filter func(id string) bool) []SearchResult {
	vi.mu.RLock()
	defer vi.mu.RUnlock()
	
	if len(vi.Vectors) == 0 {
		return nil
	}
	
	// Calculate similarities for filtered vectors
	results := make([]SearchResult, 0)
	for i, vec := range vi.Vectors {
		if !filter(vi.IDs[i]) {
			continue
		}
		
		similarity := cosineSimilarity(query, vec)
		results = append(results, SearchResult{
			ID:         vi.IDs[i],
			Similarity: similarity,
			Score:      similarity,
		})
	}
	
	// Sort by similarity (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	
	// Return top k results
	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// Size returns the number of vectors in the index
func (vi *VectorIndex) Size() int {
	vi.mu.RLock()
	defer vi.mu.RUnlock()
	return len(vi.Vectors)
}

// Clear removes all vectors from the index
func (vi *VectorIndex) Clear() {
	vi.mu.Lock()
	defer vi.mu.Unlock()
	vi.Vectors = make([][]float32, 0)
	vi.IDs = make([]string, 0)
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	magnitude := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if magnitude == 0 {
		return 0
	}
	
	return dotProduct / magnitude
}

// QuantizedVector represents a quantized vector for memory efficiency
type QuantizedVector struct {
	ID         string
	Quantized  []int8   // Quantized to int8 for memory efficiency
	Scale      float32  // Scale factor for dequantization
}

// QuantizedIndex represents a memory-efficient vector index using quantization
type QuantizedIndex struct {
	Vectors []QuantizedVector
	mu      sync.RWMutex
}

// NewQuantizedIndex creates a new quantized index
func NewQuantizedIndex() *QuantizedIndex {
	return &QuantizedIndex{
		Vectors: make([]QuantizedVector, 0),
	}
}

// quantizeVector quantizes a float32 vector to int8 for memory efficiency
func quantizeVector(vec []float32) ([]int8, float32) {
	// Find min and max for scaling
	var minVal, maxVal float32 = vec[0], vec[0]
	for _, v := range vec {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	// Calculate scale
	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1
	}
	
	// Quantize
	quantized := make([]int8, len(vec))
	for i, v := range vec {
		normalized := (v - minVal) / scale
		// Clamp to int8 range
		val := int(normalized) - 128
		if val < -128 {
			val = -128
		} else if val > 127 {
			val = 127
		}
		quantized[i] = int8(val)
	}
	
	return quantized, scale
}

// dequantizeVector converts a quantized vector back to float32
func dequantizeVector(quantized []int8, scale float32) []float32 {
	vec := make([]float32, len(quantized))
	for i, q := range quantized {
		// Convert back from centered int8 to original range
		vec[i] = (float32(q) + 128) * scale
	}
	return vec
}

// Add adds a vector to the quantized index
func (qi *QuantizedIndex) Add(id string, vector []float32) {
	qi.mu.Lock()
	defer qi.mu.Unlock()
	
	quantized, scale := quantizeVector(vector)
	qi.Vectors = append(qi.Vectors, QuantizedVector{
		ID:        id,
		Quantized: quantized,
		Scale:     scale,
	})
}

// Search performs approximate nearest neighbor search on quantized vectors
func (qi *QuantizedIndex) Search(query []float32, k int) []SearchResult {
	qi.mu.RLock()
	defer qi.mu.RUnlock()
	
	if len(qi.Vectors) == 0 {
		return nil
	}
	
	// For better accuracy, we could keep the query in float32
	// and only quantize the stored vectors
	results := make([]SearchResult, 0, len(qi.Vectors))
	
	for _, qvec := range qi.Vectors {
		// Dequantize for similarity calculation
		vec := dequantizeVector(qvec.Quantized, qvec.Scale)
		similarity := cosineSimilarity(query, vec)
		results = append(results, SearchResult{
			ID:         qvec.ID,
			Similarity: similarity,
			Score:      similarity,
		})
	}
	
	// Sort by similarity
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	
	// Return top k
	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// IndexMetadata stores metadata about the index
type IndexMetadata struct {
	Granularity string `json:"granularity"`
	Count       int    `json:"count"`
	Dimensions  int    `json:"dimensions"`
	Loaded      bool   `json:"loaded"`
}

// MarshalJSON implements json.Marshaler for IndexMetadata
func (m IndexMetadata) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Granularity string `json:"granularity"`
		Count       int    `json:"count"`
		Dimensions  int    `json:"dimensions"`
		Loaded      bool   `json:"loaded"`
	}{
		Granularity: m.Granularity,
		Count:       m.Count,
		Dimensions:  m.Dimensions,
		Loaded:      m.Loaded,
	})
}

// GetMemoryUsage estimates memory usage of the index in bytes
func (vi *VectorIndex) GetMemoryUsage() int64 {
	vi.mu.RLock()
	defer vi.mu.RUnlock()
	
	// Each float32 is 4 bytes
	vectorMemory := int64(len(vi.Vectors)) * int64(len(vi.Vectors[0])) * 4
	// Estimate string memory (rough estimate)
	idMemory := int64(0)
	for _, id := range vi.IDs {
		idMemory += int64(len(id))
	}
	
	return vectorMemory + idMemory
}

// GetMemoryUsage estimates memory usage of the quantized index
func (qi *QuantizedIndex) GetMemoryUsage() int64 {
	qi.mu.RLock()
	defer qi.mu.RUnlock()
	
	memory := int64(0)
	for _, vec := range qi.Vectors {
		// Each int8 is 1 byte, plus 4 bytes for scale, plus string length
		memory += int64(len(vec.Quantized)) + 4 + int64(len(vec.ID))
	}
	
	return memory
}