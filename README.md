<div align="center">
  <img src="https://tahsinkoc.com/banners/embrix-logo.png" alt="embrix logo" width="400">
</div>

# embrix

[![npm version](https://img.shields.io/npm/v/embrix.svg)](https://www.npmjs.com/package/embrix)
[![npm downloads](https://img.shields.io/npm/dm/embrix.svg)](https://www.npmjs.com/package/embrix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://tahsinkoc.com/papers/embrix-nodejs-embedding)

Production-ready local text embeddings using `@xenova/transformers`. Zero external API calls, runs entirely in Node.js.

## Features

- Local Execution - No API calls, runs entirely on your machine
- Two Optimized Models - MiniLM and BGE for different use cases
- Zero Dependencies - Only `@xenova/transformers` as dependency
- Type-Safe - Full TypeScript support with strict typing
- Efficient - Lazy loading, singleton pattern, batch processing
- Benchmark Tools - Built-in performance measurement utilities

## Installation

```bash
npm install embrix
```

## Quick Start

```typescript
import { Embedder, EmbeddingModel, cosineSimilarity } from 'embrix';

// Create an embedder
const embedder = new Embedder(EmbeddingModel.MiniLM);

// Generate a single embedding
const embedding = await embedder.embed("Hello, world!");
console.log(embedding.length); // 384

// Generate batch embeddings
const embeddings = await embedder.embedBatch([
  "Hello, world!",
  "Goodbye, world!"
]);

// Compare similarity
const hello = await embedder.embed("Hello!");
const goodbye = await embedder.embed("Goodbye!");
const similarity = cosineSimilarity(hello, goodbye);
console.log(`Similarity: ${similarity}`);
```

## Supported Models

| Model | Enum | Dimensions | Description |
|-------|------|------------|-------------|
| all-MiniLM-L6-v2 | `EmbeddingModel.MiniLM` | 384 | Fast and efficient, great for most use cases |
| bge-small-en-v1.5 | `EmbeddingModel.BGE` | 384 | High quality English embeddings from BAAI |

## API Reference

### Embedder Class

```typescript
import { Embedder, EmbeddingModel } from 'embrix';

const embedder = new Embedder(EmbeddingModel.MiniLM);
```

#### Properties

- `dimension: number` - Embedding dimension (384 for both models)
- `modelName: string` - Human-readable model name
- `modelType: EmbeddingModel` - The model enum value

#### Methods

##### `embed(text: string, options?): Promise<Float32Array>`

Generate an embedding for a single text.

```typescript
const vector = await embedder.embed("Your text here");
```

##### `embedBatch(texts: string[], options?): Promise<Float32Array[]>`

Generate embeddings for multiple texts efficiently.

```typescript
const vectors = await embedder.embedBatch(["Text 1", "Text 2", "Text 3"]);
```

##### `embedWithMetadata(text: string, options?): Promise<EmbeddingResult>`

Generate embedding with full metadata.

```typescript
const result = await embedder.embedWithMetadata("Your text");
console.log(result.model);      // EmbeddingModel.MiniLM
console.log(result.dimension);  // 384
console.log(result.embedding);  // Float32Array
```

### Similarity Functions

```typescript
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  manhattanDistance,
  findMostSimilar,
  findKMostSimilar
} from 'embrix';
```

#### `cosineSimilarity(a: Float32Array, b: Float32Array): number`

Calculate cosine similarity between two vectors. Range: [-1, 1].

```typescript
const similarity = cosineSimilarity(vector1, vector2);
```

#### `dotProduct(a: Float32Array, b: Float32Array): number`

Calculate dot product of two vectors.

```typescript
const product = dotProduct(vector1, vector2);
```

#### `euclideanDistance(a: Float32Array, b: Float32Array): number`

Calculate Euclidean (L2) distance between two vectors.

```typescript
const distance = euclideanDistance(vector1, vector2);
```

#### `findMostSimilar(query: Float32Array, candidates: Float32Array[])`

Find the most similar vector to a query.

```typescript
const query = await embedder.embed("search query");
const docs = await embedder.embedBatch(["doc 1", "doc 2", "doc 3"]);
const best = findMostSimilar(query, docs);
console.log(`Best match: index ${best.index}, similarity ${best.similarity}`);
```

#### `findKMostSimilar(query: Float32Array, candidates: Float32Array[], k: number)`

Find the k most similar vectors.

```typescript
const top5 = findKMostSimilar(query, docs, 5);
```

### Model Loading

```typescript
import { preloadModel, preloadAllModels, isModelLoaded, clearModelCache } from 'embrix';

// Preload a specific model
await preloadModel(EmbeddingModel.MiniLM);

// Preload all models
await preloadAllModels();

// Check if model is loaded
if (isModelLoaded(EmbeddingModel.MiniLM)) {
  // Model is ready
}

// Clear cache to free memory
clearModelCache();
```

### Benchmark Utilities

```typescript
import { runBenchmark, runAllBenchmarks, formatBenchmarkResult } from 'embrix';

// Benchmark a single model
const results = await runBenchmark(EmbeddingModel.MiniLM);
console.log(formatBenchmarkResult(results));

// Benchmark all models
const allResults = await runAllBenchmarks();
```

## CLI Benchmark

Run benchmarks from the command line:

```bash
npm run benchmark

# Options
npm run benchmark -- --model minilm
npm run benchmark -- --batch-size 50
npm run benchmark -- --help
```

## Experimental Benchmark Results

For comprehensive experimental benchmark results, including performance comparisons across different models and configurations, see the [test-embrix-experimental](https://github.com/tahsinkoc/test-embrix-experimental) repository.

## Example Output

```
============================================================
Benchmark: all-MiniLM-L6-v2
Model: minilm
Dimension: 384
============================================================

Running cold start benchmark...
  Duration: 2345.67ms

Running warm start benchmark...
  Duration: 12.34ms
  Speedup: 190.15x faster than cold start

Running batch benchmark (100 texts)...
  Total duration: 567.89ms
  Avg per embedding: 5.68ms
  Throughput: 176.09 embeddings/sec
```

## Architecture

```
embrix/
├── src/
│   ├── models.ts      # Model definitions and metadata
│   ├── loader.ts      # Lazy singleton model loader
│   ├── embedder.ts    # Core embedding class
│   ├── similarity.ts  # Vector similarity functions
│   ├── benchmark.ts   # Performance measurement utilities
│   └── index.ts       # Barrel export
├── scripts/
│   └── benchmark.ts   # CLI benchmark script
├── examples/
│   └── usage.ts       # Usage examples
├── package.json
├── tsconfig.json
└── README.md
```

## Design Decisions

### Lazy Singleton Loading

Models are loaded on-demand and cached in memory. This ensures:
- Fast subsequent calls (warm start)
- No duplicate model loads
- Memory efficiency

### Float32Array Throughout

All embeddings are returned as `Float32Array` for:
- Consistency with the underlying tensor output
- Memory efficiency vs regular arrays
- Compatibility with WebAssembly and GPU operations

### No External Dependencies

Only `@xenova/transformers` is required. This keeps the package:
- Lightweight
- Secure
- Easy to audit

## Requirements

- Node.js >= 18.0.0
- ~500MB disk space for model cache (first run)

## Citation

If you use embrix in your research, please cite:

```bibtex
@software{embrix2026,
  title = {embrix: Production-Ready Local Text Embeddings for Node.js},
  author = {Tahsin Özgür Koç},
  year = {2026},
  url = {https://github.com/tahsinkoc/embrix}
}
```

## License

MIT
