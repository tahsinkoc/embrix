import { FeatureExtractionPipeline } from '@xenova/transformers';

/**
 * @fileoverview Model definitions and metadata for embedding models.
 *
 * This module defines the supported embedding models and their configurations.
 * Each model has a specific HuggingFace path and output dimension.
 *
 * Design Decision: Using string enum for type-safe model selection
 * while allowing easy extension for future models.
 */
/**
 * Supported embedding models.
 *
 * - MiniLM: Fast, lightweight model optimized for speed
 * - BGE: Beijing Academy of AI model with strong semantic understanding
 */
declare enum EmbeddingModel {
    /** Xenova/all-MiniLM-L6-v2 - Fast and efficient, 384 dimensions */
    MiniLM = "minilm",
    /** Xenova/bge-small-en-v1.5 - High quality English embeddings, 384 dimensions */
    BGE = "bge"
}
/**
 * Configuration for an embedding model.
 */
interface ModelConfig {
    /** HuggingFace model path for @xenova/transformers */
    readonly hfPath: string;
    /** Output embedding dimension */
    readonly dimension: number;
    /** Human-readable model name */
    readonly name: string;
    /** Model description */
    readonly description: string;
    /** Maximum input sequence length */
    readonly maxLength: number;
}
/**
 * Model configurations indexed by EmbeddingModel enum.
 *
 * Design Decision: Using a const object for O(1) lookup
 * and ensuring all models have complete metadata.
 */
declare const MODEL_CONFIG: Readonly<Record<EmbeddingModel, ModelConfig>>;
/**
 * Type guard to check if a string is a valid EmbeddingModel.
 */
declare function isValidModel(model: string): model is EmbeddingModel;
/**
 * Get model configuration with validation.
 * Throws an error if the model is not found.
 */
declare function getModelConfig(model: EmbeddingModel): ModelConfig;
/**
 * Get all supported model names.
 */
declare function getSupportedModels(): readonly EmbeddingModel[];

/**
 * @fileoverview Lazy singleton model loader for embedding pipelines.
 *
 * This module implements a lazy loading pattern with in-memory caching
 * to ensure models are loaded only once during the application lifecycle.
 *
 * Design Decision: Using Map cache instead of class-based singleton
 * for simpler API and better tree-shaking support.
 */

/**
 * Load an embedding model pipeline with caching.
 *
 * This function implements lazy loading with singleton pattern:
 * - First call: Downloads and caches the model
 * - Subsequent calls: Returns cached model immediately
 *
 * Thread-safe: Concurrent calls for the same model will share the same
 * loading promise, preventing duplicate downloads.
 *
 * @param model - The embedding model to load
 * @returns Promise resolving to the feature extraction pipeline
 * @throws Error if model fails to load
 *
 * @example
 * ```typescript
 * const pipeline = await loadModel(EmbeddingModel.MiniLM);
 * // Subsequent calls return cached pipeline
 * const samePipeline = await loadModel(EmbeddingModel.MiniLM);
 * ```
 */
declare function loadModel(model: EmbeddingModel): Promise<FeatureExtractionPipeline>;
/**
 * Check if a model is already loaded and cached.
 *
 * @param model - The embedding model to check
 * @returns true if model is cached and ready to use
 */
declare function isModelLoaded(model: EmbeddingModel): boolean;
/**
 * Clear the model cache.
 *
 * Useful for freeing memory or forcing model reload.
 * Warning: This will require re-downloading models on next use.
 */
declare function clearModelCache(): void;
/**
 * Get list of currently loaded models.
 */
declare function getLoadedModels(): EmbeddingModel[];
/**
 * Preload a model into cache.
 *
 * Useful for warming up models before first use.
 *
 * @param model - The embedding model to preload
 * @returns Promise that resolves when model is loaded
 *
 * @example
 * ```typescript
 * // Preload model during application startup
 * await preloadModel(EmbeddingModel.MiniLM);
 * ```
 */
declare function preloadModel(model: EmbeddingModel): Promise<void>;
/**
 * Preload all supported models.
 *
 * @returns Promise that resolves when all models are loaded
 */
declare function preloadAllModels(): Promise<void>;

/**
 * @fileoverview Core embedding functionality.
 *
 * This module provides the Embedder class for generating text embeddings
 * using the loaded model pipelines.
 *
 * Design Decision: Using a class-based API for better encapsulation
 * and to allow multiple embedder instances with different models.
 */

/**
 * Options for embedding generation.
 */
interface EmbedOptions {
    /** Whether to normalize the output embedding (default: true) */
    normalize?: boolean;
    /** Pooling strategy to use (default: "mean") */
    pooling?: "mean" | "cls";
}
/**
 * Result of an embedding operation with metadata.
 */
interface EmbeddingResult {
    /** The embedding vector */
    embedding: Float32Array;
    /** The model used for embedding */
    model: EmbeddingModel;
    /** Dimension of the embedding */
    dimension: number;
}
/**
 * Core class for generating text embeddings.
 *
 * The Embedder class provides a clean API for generating embeddings
 * from text using pre-trained transformer models.
 *
 * @example
 * ```typescript
 * const embedder = new Embedder(EmbeddingModel.MiniLM);
 *
 * // Single embedding
 * const vector = await embedder.embed("Hello world");
 *
 * // Batch embedding
 * const vectors = await embedder.embedBatch(["Hello", "World"]);
 * ```
 */
declare class Embedder {
    private readonly model;
    private readonly config;
    /**
     * Create a new Embedder instance.
     *
     * @param model - The embedding model to use
     * @throws Error if the model is not supported
     */
    constructor(model: EmbeddingModel);
    /**
     * Get the dimension of embeddings produced by this model.
     *
     * @returns The embedding dimension
     */
    get dimension(): number;
    /**
     * Get the model name.
     *
     * @returns The human-readable model name
     */
    get modelName(): string;
    /**
     * Get the model enum value.
     *
     * @returns The EmbeddingModel enum value
     */
    get modelType(): EmbeddingModel;
    /**
     * Generate an embedding for a single text.
     *
     * @param text - The text to embed
     * @param options - Optional embedding configuration
     * @returns Promise resolving to the embedding vector
     * @throws Error if embedding generation fails
     *
     * @example
     * ```typescript
     * const embedder = new Embedder(EmbeddingModel.MiniLM);
     * const vector = await embedder.embed("Hello world");
     * console.log(vector.length); // 384
     * ```
     */
    embed(text: string, options?: EmbedOptions): Promise<Float32Array>;
    /**
     * Generate embeddings for multiple texts.
     *
     * This method is more efficient than calling embed() multiple times
     * as it processes all texts in a single batch.
     *
     * @param texts - Array of texts to embed
     * @param options - Optional embedding configuration
     * @returns Promise resolving to array of embedding vectors
     * @throws Error if embedding generation fails
     *
     * @example
     * ```typescript
     * const embedder = new Embedder(EmbeddingModel.MiniLM);
     * const vectors = await embedder.embedBatch(["Hello", "World"]);
     * console.log(vectors.length); // 2
     * console.log(vectors[0].length); // 384
     * ```
     */
    embedBatch(texts: string[], options?: EmbedOptions): Promise<Float32Array[]>;
    /**
     * Generate an embedding with full result metadata.
     *
     * @param text - The text to embed
     * @param options - Optional embedding configuration
     * @returns Promise resolving to embedding result with metadata
     */
    embedWithMetadata(text: string, options?: EmbedOptions): Promise<EmbeddingResult>;
    /**
     * Check if the model for this embedder is loaded.
     *
     * @returns true if the model is loaded and ready
     */
    isReady(): boolean;
    /**
     * Convert a Tensor to Float32Array.
     *
     * Handles different tensor data types and shapes.
     */
    private tensorToFloat32;
    /**
     * Extract vectors from batch output tensor.
     *
     * Handles the tensor shape from batch processing.
     */
    private extractBatchVectors;
    /**
     * Validate that an embedding has the expected dimension.
     *
     * @param vector - The embedding vector to validate
     * @throws Error if dimension doesn't match expected
     */
    private validateDimension;
}
/**
 * Factory function to create an Embedder instance.
 *
 * Convenience function for creating embedders without using new.
 *
 * @param model - The embedding model to use
 * @returns A new Embedder instance
 *
 * @example
 * ```typescript
 * const embedder = createEmbedder(EmbeddingModel.MiniLM);
 * const vector = await embedder.embed("Hello world");
 * ```
 */
declare function createEmbedder(model: EmbeddingModel): Embedder;

/**
 * @fileoverview Vector similarity and distance functions.
 *
 * This module provides efficient implementations of common similarity
 * and distance metrics for comparing embedding vectors.
 *
 * Design Decision: All functions work with Float32Array for consistency
 * with the embedding output and optimal memory usage.
 */
/**
 * Calculate the dot product of two vectors.
 *
 * The dot product is the sum of element-wise products.
 * For normalized vectors, this equals cosine similarity.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The dot product
 * @throws Error if vectors have different lengths
 *
 * @example
 * ```typescript
 * const a = new Float32Array([1, 2, 3]);
 * const b = new Float32Array([4, 5, 6]);
 * const product = dotProduct(a, b); // 1*4 + 2*5 + 3*6 = 32
 * ```
 */
declare function dotProduct(a: Float32Array, b: Float32Array): number;
/**
 * Calculate the cosine similarity between two vectors.
 *
 * Cosine similarity measures the cosine of the angle between two vectors.
 * Range: [-1, 1] where 1 means identical direction, -1 means opposite.
 *
 * Formula: cos(θ) = (A · B) / (||A|| * ||B||)
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The cosine similarity (-1 to 1)
 * @throws Error if vectors have different lengths or zero magnitude
 *
 * @example
 * ```typescript
 * const a = new Float32Array([1, 0, 0]);
 * const b = new Float32Array([0, 1, 0]);
 * const similarity = cosineSimilarity(a, b); // 0 (orthogonal)
 * ```
 */
declare function cosineSimilarity(a: Float32Array, b: Float32Array): number;
/**
 * Calculate the Euclidean distance between two vectors.
 *
 * Euclidean distance is the straight-line distance between two points
 * in Euclidean space. Lower values indicate more similar vectors.
 *
 * Formula: d(A, B) = sqrt(Σ(Ai - Bi)²)
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The Euclidean distance (0 to infinity)
 * @throws Error if vectors have different lengths
 *
 * @example
 * ```typescript
 * const a = new Float32Array([0, 0, 0]);
 * const b = new Float32Array([1, 1, 1]);
 * const distance = euclideanDistance(a, b); // sqrt(3) ≈ 1.732
 * ```
 */
declare function euclideanDistance(a: Float32Array, b: Float32Array): number;
/**
 * Calculate the squared Euclidean distance between two vectors.
 *
 * More efficient than euclideanDistance when only comparing distances
 * (avoids the sqrt operation).
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The squared Euclidean distance
 * @throws Error if vectors have different lengths
 */
declare function euclideanDistanceSquared(a: Float32Array, b: Float32Array): number;
/**
 * Calculate the Manhattan distance between two vectors.
 *
 * Manhattan distance is the sum of absolute differences.
 * Also known as L1 distance or city block distance.
 *
 * Formula: d(A, B) = Σ|Ai - Bi|
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The Manhattan distance
 * @throws Error if vectors have different lengths
 */
declare function manhattanDistance(a: Float32Array, b: Float32Array): number;
/**
 * Calculate the magnitude (L2 norm) of a vector.
 *
 * @param vector - The input vector
 * @returns The magnitude
 */
declare function magnitude(vector: Float32Array): number;
/**
 * Normalize a vector to unit length.
 *
 * Returns a new vector with the same direction but magnitude 1.
 *
 * @param vector - The input vector
 * @returns A new normalized vector
 * @throws Error if vector has zero magnitude
 */
declare function normalize(vector: Float32Array): Float32Array;
/**
 * Find the most similar vector to a query from a set of candidates.
 *
 * @param query - The query vector
 * @param candidates - Array of candidate vectors
 * @returns Index and similarity score of the most similar candidate
 * @throws Error if candidates array is empty
 *
 * @example
 * ```typescript
 * const query = await embedder.embed("hello");
 * const docs = await embedder.embedBatch(["hi", "goodbye", "greetings"]);
 * const { index, similarity } = findMostSimilar(query, docs);
 * console.log(`Best match at index ${index} with similarity ${similarity}`);
 * ```
 */
declare function findMostSimilar(query: Float32Array, candidates: Float32Array[]): {
    index: number;
    similarity: number;
};
/**
 * Find the k most similar vectors to a query from a set of candidates.
 *
 * @param query - The query vector
 * @param candidates - Array of candidate vectors
 * @param k - Number of results to return
 * @returns Array of indices and similarity scores, sorted by similarity descending
 *
 * @example
 * ```typescript
 * const query = await embedder.embed("hello");
 * const docs = await embedder.embedBatch(["hi", "goodbye", "greetings", "farewell"]);
 * const top2 = findKMostSimilar(query, docs, 2);
 * // Returns top 2 most similar documents
 * ```
 */
declare function findKMostSimilar(query: Float32Array, candidates: Float32Array[], k: number): Array<{
    index: number;
    similarity: number;
}>;

/**
 * @fileoverview Benchmark utilities for measuring embedding performance.
 *
 * This module provides tools to measure and analyze embedding generation
 * performance, including cold/warm start times and batch throughput.
 *
 * Design Decision: Using process.hrtime.bigint() for nanosecond precision
 * timing without external dependencies.
 */

/**
 * Result of a single benchmark measurement.
 */
interface BenchmarkResult {
    /** Operation name */
    operation: string;
    /** Time taken in milliseconds */
    durationMs: number;
    /** Number of embeddings generated */
    embeddingCount: number;
    /** Average time per embedding in milliseconds */
    avgTimePerEmbeddingMs: number;
    /** Embeddings per second */
    throughput: number;
}
/**
 * Complete benchmark suite results.
 */
interface BenchmarkSuiteResult {
    /** Model used for benchmarking */
    model: EmbeddingModel;
    /** Model name */
    modelName: string;
    /** Embedding dimension */
    dimension: number;
    /** Cold start time (first embed call) */
    coldStart: BenchmarkResult;
    /** Warm start time (second embed call) */
    warmStart: BenchmarkResult;
    /** Batch embedding results */
    batchBenchmark: BenchmarkResult;
    /** Total benchmark duration in milliseconds */
    totalDurationMs: number;
    /** Timestamp of benchmark */
    timestamp: string;
}
/**
 * Options for benchmark configuration.
 */
interface BenchmarkOptions {
    /** Number of texts for batch benchmark (default: 100) */
    batchSize?: number;
    /** Text to use for single embedding benchmarks */
    sampleText?: string;
    /** Whether to log progress to console */
    verbose?: boolean;
}
/**
 * Run a complete benchmark suite for an embedding model.
 *
 * Measures:
 * - Cold start time: First embedding call (includes model loading)
 * - Warm start time: Second embedding call (model already loaded)
 * - Batch throughput: Time to embed multiple texts
 *
 * @param model - The embedding model to benchmark
 * @param options - Benchmark configuration options
 * @returns Complete benchmark results
 *
 * @example
 * ```typescript
 * const results = await runBenchmark(EmbeddingModel.MiniLM);
 * console.log(`Cold start: ${results.coldStart.durationMs}ms`);
 * console.log(`Throughput: ${results.batchBenchmark.throughput} embeddings/sec`);
 * ```
 */
declare function runBenchmark(model: EmbeddingModel, options?: BenchmarkOptions): Promise<BenchmarkSuiteResult>;
/**
 * Run benchmarks for all supported models.
 *
 * @param options - Benchmark configuration options
 * @returns Array of benchmark results for each model
 */
declare function runAllBenchmarks(options?: BenchmarkOptions): Promise<BenchmarkSuiteResult[]>;
/**
 * Format benchmark results as a printable string.
 *
 * @param result - Benchmark suite result
 * @returns Formatted string representation
 */
declare function formatBenchmarkResult(result: BenchmarkSuiteResult): string;
/**
 * Compare benchmark results between two models.
 *
 * @param result1 - First model's benchmark results
 * @param result2 - Second model's benchmark results
 * @returns Comparison summary string
 */
declare function compareBenchmarks(result1: BenchmarkSuiteResult, result2: BenchmarkSuiteResult): string;

export { type BenchmarkOptions, type BenchmarkResult, type BenchmarkSuiteResult, type EmbedOptions, Embedder, EmbeddingModel, type EmbeddingResult, MODEL_CONFIG, type ModelConfig, clearModelCache, compareBenchmarks, cosineSimilarity, createEmbedder, dotProduct, euclideanDistance, euclideanDistanceSquared, findKMostSimilar, findMostSimilar, formatBenchmarkResult, getLoadedModels, getModelConfig, getSupportedModels, isModelLoaded, isValidModel, loadModel, magnitude, manhattanDistance, normalize, preloadAllModels, preloadModel, runAllBenchmarks, runBenchmark };
