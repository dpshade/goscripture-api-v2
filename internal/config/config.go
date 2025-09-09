package config

// Config holds the application configuration
type Config struct {
	Port      string
	ModelPath string
	DataDir   string
	Debug     bool
}

// ArweaveURLs contains the URLs for pre-computed embeddings and text data
var ArweaveURLs = struct {
	Verses              string
	VersesUncompressed  string
	VerseText          string
	Chapters           string
	ChaptersUncompressed string
	ChapterText        string
}{
	Verses:              "https://arweave.net/DdKgzVD1zJDlFqcPKnOf620EviiU5LoGDqxrv5wxxUY",
	VersesUncompressed:  "https://arweave.net/uW8W0WE37IsmZuuH5NJxfrDLWjf_vFFK5v4MmzL2Gco",
	VerseText:          "https://arweave.net/daKtqqHpLRnAWCNEWY8Q92NwSyJxWbm7WFDE3ut_BuM",
	Chapters:           "https://arweave.net/RLfnXY9-kD9McjalEhGgcK3ZZH9G_dXcfh6unygV1s8",
	ChaptersUncompressed: "https://arweave.net/JyOVgeD6IlWnIg8e24cPQgEidAADriGexHbhZ0Hs6Fo",
	ChapterText:        "https://arweave.net/daKtqqHpLRnAWCNEWY8Q92NwSyJxWbm7WFDE3ut_BuM",
}

// ModelConfig contains model-specific configuration
var ModelConfig = struct {
	ModelID      string
	Dimensions   int
	QueryPrefix  string
	DocumentPrefix string
}{
	ModelID:       "onnx-community/embeddinggemma-300m-ONNX",
	Dimensions:    128, // Using 128D Matryoshka truncation
	QueryPrefix:   "task: search result | query: ",
	DocumentPrefix: "title: none | text: ",
}