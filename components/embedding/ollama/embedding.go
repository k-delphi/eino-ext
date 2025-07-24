package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/ollama/ollama/api"
)

// Set with env: os.Setenv("QIANFAN_ACCESS_KEY", "your_iam_ak") or with env file

type Duration struct {
	time.Duration
}
type EmbeddingConfig struct {
	BaseURL    string                 `json:"base_url"`
	Model      string                 `json:"model"`       // 必填，模型名称
	HTTPClient *http.Client           `json:"http_client"` // Ollama 服务地址，默认 http://localhost:11434
	Timeout    time.Duration          `json:"timeout"`     // HTTP 请求超时时间
	Truncate   bool                   `json:"truncate"`    // 是否截断输入，默认 true
	Options    map[string]interface{} `json:"options"`     // 可选参数，如 temperature
	KeepAlive  *Duration              `json:"keep_alive"`  // 模型保活时间，默认 "5m"
}

type Embedder struct {
	cli *EmbeddingClient
}

type EmbeddingClient struct {
	cli    *api.Client
	config *EmbeddingConfig
}

func DefaultConfig() *EmbeddingConfig {
	return &EmbeddingConfig{
		BaseURL:    "http://localhost:11434",
		Model:      "mxbai-embed-large",
		HTTPClient: &http.Client{Timeout: time.Second * 30},
		Timeout:    30 * time.Second,
		Truncate:   true,
		Options:    make(map[string]interface{}),
		KeepAlive:  &Duration{Duration: 5 * time.Minute},
	}
}

func NewEmbedder(ctx context.Context, config *EmbeddingConfig) (*Embedder, error) {
	if config == nil {
		config = DefaultConfig()
	}
	baseURL, err := url.Parse(config.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	cli := api.NewClient(baseURL, httpClient)

	client := &EmbeddingClient{
		cli:    cli,
		config: config,
	}

	return &Embedder{
		cli: client,
	}, nil
}

func (e *Embedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) (
	embeddings [][]float64, err error) {
	embeddings = make([][]float64, len(texts))
	for i, text := range texts {
		resp, err := e.cli.cli.Embeddings(ctx, &api.EmbeddingRequest{
			Model:     e.cli.config.Model,
			Prompt:    text,
			Options:   e.cli.config.Options,
			KeepAlive: (*api.Duration)(e.cli.config.KeepAlive),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get embeddings: %w", err)
		}
		embeddings[i] = resp.Embedding
	}
	return embeddings, nil
}

const typ = "Ollama"

func (e *Embedder) GetType() string {
	return typ
}

func (e *Embedder) IsCallbacksEnabled() bool {
	return true
}
