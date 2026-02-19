# embrix v1.0.0

Production-ready local text embeddings using `@xenova/transformers`. Zero external API calls, runs entirely in Node.js.

## Installation

```bash
npm install embrix
```

## Features

- **Local Execution** - No API calls, runs entirely on your machine
- **Two Optimized Models** - MiniLM and BGE for different use cases
- **Zero Dependencies** - Only `@xenova/transformers` as dependency
- **Type-Safe** - Full TypeScript support with strict typing
- **Efficient** - Lazy loading, singleton pattern, batch processing
- **Benchmark Tools** - Built-in performance measurement utilities

## Quick Start

```typescript
import { Embedder, EmbeddingModel, cosineSimilarity } from 'embrix';

const embedder = new Embedder(EmbeddingModel.MiniLM);

// Generate embeddings
const embedding = await embedder.embed("Hello, world!");
console.log(embedding.length); // 384

// Compare similarity
const hello = await embedder.embed("Hello!");
const goodbye = await embedder.embed("Goodbye!");
const similarity = cosineSimilarity(hello, goodbye);
```

## Supported Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | Fast and efficient, great for most use cases |
| bge-small-en-v1.5 | 384 | High quality English embeddings from BAAI |

## Links

- [npm Package](https://www.npmjs.com/package/embrix)
- [GitHub Repository](https://github.com/tahsinkoc/embrix)

## Requirements

- Node.js >= 18.0.0
- ~500MB disk space for model cache (first run)

## License

MIT
