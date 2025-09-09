package api

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/dpshade/goscriptureapi/internal/search"
	"github.com/labstack/echo/v4"
	"github.com/rs/zerolog/log"
)

// Handler handles API requests
type Handler struct {
	search *search.SearchService
}

// NewHandler creates a new API handler
func NewHandler(searchService *search.SearchService) *Handler {
	return &Handler{
		search: searchService,
	}
}

// Health handles health check requests
func (h *Handler) Health(c echo.Context) error {
	return c.JSON(http.StatusOK, map[string]string{
		"status": "healthy",
	})
}

// Status handles status requests
func (h *Handler) Status(c echo.Context) error {
	status := h.search.GetStatus()
	return c.JSON(http.StatusOK, status)
}

// SearchRequest represents a search request
type SearchRequest struct {
	Query       string                `json:"query"`
	Options     search.SearchOptions  `json:"options,omitempty"`
	Granularity string               `json:"granularity,omitempty"`
	K           int                  `json:"k,omitempty"`
	Book        string               `json:"book,omitempty"`
	Chapter     string               `json:"chapter,omitempty"`
	Verse       string               `json:"verse,omitempty"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	Query   string                 `json:"query"`
	Results []BibleVerseResult     `json:"results"`
	Count   int                   `json:"count"`
	Status  string                `json:"status"`
}

// BibleVerseResult represents a Bible verse search result
type BibleVerseResult struct {
	Book       string                 `json:"book"`
	Chapter    int                   `json:"chapter"`
	VerseNum   int                   `json:"verseNum"`
	Text       string                `json:"text"`
	SearchMeta map[string]interface{} `json:"_searchMeta,omitempty"`
}

// Search handles search requests (both GET and POST)
func (h *Handler) Search(c echo.Context) error {
	var req SearchRequest
	
	// Handle GET request with query parameters
	if c.Request().Method == "GET" {
		req.Query = c.QueryParam("q")
		if req.Query == "" {
			req.Query = c.QueryParam("query")
		}
		
		// Parse optional parameters
		if k := c.QueryParam("k"); k != "" {
			if kVal, err := strconv.Atoi(k); err == nil {
				req.K = kVal
			}
		}
		
		req.Book = c.QueryParam("book")
		req.Chapter = c.QueryParam("chapter")
		req.Verse = c.QueryParam("verse")
		req.Granularity = c.QueryParam("granularity")
		
	} else {
		// Handle POST request with JSON body
		if err := c.Bind(&req); err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{
				"error": "Invalid request body",
			})
		}
	}

	// Parse query to extract filters
	query, filters := parseQuery(req.Query)

	// Merge request options with parsed filters
	options := search.SearchOptions{
		Book:        coalesce(req.Book, filters.Book, req.Options.Book),
		Chapter:     coalesce(req.Chapter, filters.Chapter, req.Options.Chapter),
		Verse:       coalesce(req.Verse, filters.Verse, req.Options.Verse),
		Granularity: coalesce(req.Granularity, req.Options.Granularity, "verse"),
		K:           maxInt(req.K, req.Options.K, 10),
	}

	// Perform search
	results, err := h.search.Search(query, options)
	if err != nil {
		log.Error().Err(err).Msg("Search failed")
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Search failed",
			"details": err.Error(),
		})
	}

	// Convert results to Bible verse format
	verses := make([]BibleVerseResult, 0, len(results))
	for _, result := range results {
		verse := BibleVerseResult{
			Book:     result.Chunk.Meta.Book,
			Chapter:  result.Chunk.Meta.Chapter,
			VerseNum: result.Chunk.Meta.VerseNum,
			Text:     result.Chunk.Text,
			SearchMeta: map[string]interface{}{
				"similarity": result.Similarity,
				"score":      result.Score,
				"reference":  result.Chunk.Meta.Reference,
			},
		}
		verses = append(verses, verse)
	}

	response := SearchResponse{
		Query:   req.Query,
		Results: verses,
		Count:   len(verses),
		Status:  "success",
	}

	return c.JSON(http.StatusOK, response)
}

// EmbedRequest represents an embedding request
type EmbedRequest struct {
	Text string `json:"text"`
	Type string `json:"type,omitempty"` // "query" or "document"
}

// EmbedResponse represents an embedding response
type EmbedResponse struct {
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
}

// Embed handles embedding generation requests
func (h *Handler) Embed(c echo.Context) error {
	var req EmbedRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Invalid request body",
		})
	}

	if req.Text == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Text is required",
		})
	}

	// This endpoint would typically generate embeddings using the model
	// For now, it returns a message indicating the feature is planned
	return c.JSON(http.StatusNotImplemented, map[string]string{
		"message": "Direct embedding generation will be available when ONNX runtime is integrated",
		"status": "planned",
	})
}

// parseQuery parses a search query to extract filters
func parseQuery(query string) (string, search.SearchOptions) {
	filters := search.SearchOptions{}
	parts := strings.Fields(query)
	semanticParts := []string{}

	for _, part := range parts {
		if strings.Contains(part, ":") {
			kv := strings.SplitN(part, ":", 2)
			if len(kv) == 2 {
				key := strings.ToLower(kv[0])
				value := kv[1]

				switch key {
				case "book":
					filters.Book = value
				case "chapter":
					filters.Chapter = value
				case "verse":
					filters.Verse = value
				default:
					semanticParts = append(semanticParts, part)
				}
			} else {
				semanticParts = append(semanticParts, part)
			}
		} else {
			semanticParts = append(semanticParts, part)
		}
	}

	return strings.Join(semanticParts, " "), filters
}

// Helper functions
func coalesce(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func maxInt(values ...int) int {
	max := 0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}