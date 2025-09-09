package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/dpshade/goscriptureapi/internal/api"
	"github.com/dpshade/goscriptureapi/internal/config"
	"github.com/dpshade/goscriptureapi/internal/embeddings"
	"github.com/dpshade/goscriptureapi/internal/search"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Parse command line flags
	port := flag.String("port", "8080", "Port to listen on")
	modelPath := flag.String("model", "", "Path to ONNX model file (optional, will download if not provided)")
	dataDir := flag.String("data", "./data", "Directory to store cached data")
	debug := flag.Bool("debug", false, "Enable debug logging")
	flag.Parse()

	// Setup logging
	zerolog.TimeFieldFormat = time.RFC3339
	if *debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	// Create configuration
	cfg := &config.Config{
		Port:      *port,
		ModelPath: *modelPath,
		DataDir:   *dataDir,
		Debug:     *debug,
	}

	// Initialize embedding service
	log.Info().Msg("Initializing EmbeddingGemma service...")
	embeddingService, err := embeddings.NewEmbeddingService(cfg)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize embedding service")
	}

	// Initialize search index
	log.Info().Msg("Initializing search index...")
	searchService, err := search.NewSearchService(embeddingService, cfg)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize search service")
	}

	// Preload indices in background
	go func() {
		log.Info().Msg("Preloading verse embeddings...")
		if err := searchService.PreloadGranularity("verse"); err != nil {
			log.Error().Err(err).Msg("Failed to preload verse embeddings")
		} else {
			log.Info().Msg("Verse embeddings loaded successfully")
		}

		log.Info().Msg("Preloading chapter embeddings...")
		if err := searchService.PreloadGranularity("chapter"); err != nil {
			log.Error().Err(err).Msg("Failed to preload chapter embeddings")
		} else {
			log.Info().Msg("Chapter embeddings loaded successfully")
		}
	}()

	// Setup Echo server
	e := echo.New()
	e.HideBanner = true
	e.HidePort = true

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"},
		AllowMethods: []string{echo.GET, echo.POST, echo.OPTIONS},
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept},
	}))

	// API handler
	apiHandler := api.NewHandler(searchService)

	// Routes
	e.GET("/health", apiHandler.Health)
	e.GET("/status", apiHandler.Status)
	e.GET("/search", apiHandler.Search)   // Support GET for search
	e.POST("/search", apiHandler.Search)  // Keep POST support
	e.POST("/embed", apiHandler.Embed)

	// Start server in goroutine
	go func() {
		log.Info().Str("port", cfg.Port).Msg("Starting HTTP server")
		if err := e.Start(fmt.Sprintf(":%s", cfg.Port)); err != nil {
			log.Error().Err(err).Msg("Server error")
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	<-quit

	log.Info().Msg("Shutting down server...")
	if err := e.Close(); err != nil {
		log.Error().Err(err).Msg("Error closing server")
	}
}