/**
 * @fileoverview Main entry point for the embrix embedding library.
 * 
 * This module exports all public APIs for generating text embeddings
 * using local transformer models.
 * 
 * @packageDocumentation
 * 
 * @example
 * ```typescript
 * import { Embedder, EmbeddingModel, cosineSimilarity } from 'embrix';
 * 
 * // Create an embedder
 * const embedder = new Embedder(EmbeddingModel.MiniLM);
 * 
 * // Generate embeddings
 * const hello = await embedder.embed("Hello, world!");
 * const goodbye = await embedder.embed("Goodbye, world!");
 * 
 * // Compare similarity
 * const similarity = cosineSimilarity(hello, goodbye);
 * console.log(`Similarity: ${similarity}`);
 * ```
 */

// ============================================================================
// Model Definitions
// ============================================================================

export {
  EmbeddingModel,
  MODEL_CONFIG,
  isValidModel,
  getModelConfig,
  getSupportedModels,
  type ModelConfig
} from "./models";

// ============================================================================
// Model Loader
// ============================================================================

export {
  loadModel,
  isModelLoaded,
  clearModelCache,
  getLoadedModels,
  preloadModel,
  preloadAllModels
} from "./loader";

// ============================================================================
// Embedder Class
// ============================================================================

export {
  Embedder,
  createEmbedder,
  type EmbedOptions,
  type EmbeddingResult
} from "./embedder";

// ============================================================================
// Similarity Functions
// ============================================================================

export {
  dotProduct,
  cosineSimilarity,
  euclideanDistance,
  euclideanDistanceSquared,
  manhattanDistance,
  magnitude,
  normalize,
  findMostSimilar,
  findKMostSimilar
} from "./similarity";

// ============================================================================
// Benchmark Utilities
// ============================================================================

export {
  runBenchmark,
  runAllBenchmarks,
  formatBenchmarkResult,
  compareBenchmarks,
  type BenchmarkResult,
  type BenchmarkSuiteResult,
  type BenchmarkOptions
} from "./benchmark";
