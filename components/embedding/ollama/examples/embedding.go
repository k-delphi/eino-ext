package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/compose"
	callbacksHelper "github.com/cloudwego/eino/utils/callbacks"

	"github.com/cloudwego/eino-ext/components/embedding/ollama"
)

func main() {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	ctx := context.Background()

	embedder, err := ollama.NewEmbedder(ctx, &ollama.EmbeddingConfig{
		BaseURL:  baseURL,
		Model:    "nomic-embed-text",
		Truncate: true,
		Timeout:  0,
		Options: map[string]any{
			"temperature": 0.2,
		},
		KeepAlive: &ollama.Duration{Duration: 5 * time.Minute},
	})
	if err != nil {
		log.Fatalf("NewEmbedder of ollama failed, err=%v", err)
	}

	log.Printf("===== call Embedder directly =====")
	vectors, err := embedder.EmbedStrings(ctx, []string{"hello", "how are you"})
	if err != nil {
		log.Fatalf("EmbedStrings failed, err=%v", err)
	}
	log.Printf("vectors : %v", vectors)

	handlerHelper := &callbacksHelper.EmbeddingCallbackHandler{
		OnStart: func(ctx context.Context, runInfo *callbacks.RunInfo, input *embedding.CallbackInput) context.Context {
			log.Printf("input access, len: %v, content: %s\n", len(input.Texts), input.Texts)
			return ctx
		},
		OnEnd: func(ctx context.Context, runInfo *callbacks.RunInfo, output *embedding.CallbackOutput) context.Context {
			log.Printf("output finished, len: %v\n", len(output.Embeddings))
			return ctx
		},
	}

	handler := callbacksHelper.NewHandlerHelper().
		Embedding(handlerHelper).
		Handler()

	chain := compose.NewChain[[]string, [][]float64]()
	chain.AppendEmbedding(embedder)

	runnable, err := chain.Compile(ctx)
	if err != nil {
		log.Fatalf("chain Compile failed, err=%v", err)
	}

	log.Printf("===== call Embedder in Chain =====")
	vectors, err = runnable.Invoke(ctx, []string{"hello", "how are you"},
		compose.WithCallbacks(handler))
	if err != nil {
		log.Fatalf("Invoke of runnable failed, err=%v", err)
	}

	log.Printf("vectors in chain: %v", vectors)
}
