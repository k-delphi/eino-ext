<<<<<<< HEAD
/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

=======
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

<<<<<<< HEAD
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
=======
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
	"github.com/cloudwego/eino/components/embedding"
	"github.com/ollama/ollama/api"
)

<<<<<<< HEAD
var (
	defaultBaseUrl = "http://localhost:11434"
)

const (
	TotalDuration   = "total_duration" // in milliseconds
	LoadDuration    = "load_duration"  // in milliseconds
	PromptEvalCount = "prompt_eval_count"
)

type EmbeddingConfig struct {
	// Timeout specifies the maximum duration to wait for API responses
	// If HTTPClient is set, Timeout will not be used.
	// Optional. Default: no timeout
	Timeout time.Duration `json:"timeout"`

	// HTTPClient specifies the client to send HTTP requests.
	// If HTTPClient is set, Timeout will not be used.
	// Optional. Default &http.Client{Timeout: Timeout}
	HTTPClient *http.Client `json:"http_client"`

	// BaseURL specifies the Ollama service endpoint URL
	// Format: http(s)://host:port
	// Optional. Default: "http://localhost:11434"
	BaseURL string `json:"base_url"`

	// Model specifies the ID of the model to use for embedding generation
	// Required
	Model string `json:"model"`

	// Truncate specifies whether to truncate text to model's maximum context length
	// When set to true, if text to embed exceeds the model's maximum context length,
	// a call to EmbedStrings will return an error
	// Optional.
	Truncate *bool `json:"truncate,omitempty"`

	// KeepAlive controls how long the model will stay loaded in memory following this request.
	// Optional. Default 5 minutes
	KeepAlive *time.Duration `json:"keep_alive,omitempty"`

	// Options lists model-specific options.
	// Optional
	Options map[string]any `json:"options,omitempty"`
}

var _ embedding.Embedder = (*Embedder)(nil)

type Embedder struct {
	cli  *api.Client
	conf *EmbeddingConfig
=======
// Set with env: os.Setenv("OLLAMA_BASE_URL", "your_ollama_base_url") or with env file
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
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
}

func NewEmbedder(ctx context.Context, config *EmbeddingConfig) (*Embedder, error) {
	if config == nil {
<<<<<<< HEAD
		return nil, fmt.Errorf("embedding config must not be nil")
	}

	if len(config.BaseURL) == 0 {
		config.BaseURL = defaultBaseUrl
	}

	var httpClient *http.Client
	if config.HTTPClient != nil {
		httpClient = config.HTTPClient
	} else {
		httpClient = &http.Client{Timeout: config.Timeout}
	}

=======
		config = DefaultConfig()
	}
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
	baseURL, err := url.Parse(config.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}
<<<<<<< HEAD
	cli := api.NewClient(baseURL, httpClient)
	return &Embedder{
		cli:  cli,
		conf: config,
=======

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
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
	}, nil
}

func (e *Embedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) (
	embeddings [][]float64, err error) {
<<<<<<< HEAD
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	req := &api.EmbedRequest{
		Model:    e.conf.Model,
		Input:    texts,
		Truncate: e.conf.Truncate,
		Options:  e.conf.Options,
	}
	if e.conf.KeepAlive != nil {
		req.KeepAlive = &api.Duration{Duration: *e.conf.KeepAlive}
	}

	options := embedding.GetCommonOptions(&embedding.Options{
		Model: &e.conf.Model,
	}, opts...)

	conf := &embedding.Config{
		Model: *options.Model,
	}

	ctx = callbacks.EnsureRunInfo(ctx, e.GetType(), components.ComponentOfEmbedding)
	ctx = callbacks.OnStart(ctx, &embedding.CallbackInput{
		Texts:  texts,
		Config: conf,
	})

	resp, err := e.cli.Embed(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("[Ollama] EmbedStrings error: %v", err)
	}

	// Convert [][]float32 to [][]float64
	result := make([][]float64, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		result[i] = make([]float64, len(emb))
		for j, v := range emb {
			result[i][j] = float64(v)
		}
	}

	extra := map[string]any{
		TotalDuration:   resp.TotalDuration,
		LoadDuration:    resp.LoadDuration,
		PromptEvalCount: resp.PromptEvalCount,
	}

	callbacks.OnEnd(ctx, &embedding.CallbackOutput{
		Embeddings: result,
		Config:     conf,
		Extra:      extra,
	})

	return result, nil
=======
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
>>>>>>> 66abd46c9cf766c53b9097e98b7c7570ad85a387
}

const typ = "Ollama"

func (e *Embedder) GetType() string {
	return typ
}

func (e *Embedder) IsCallbacksEnabled() bool {
	return true
}
