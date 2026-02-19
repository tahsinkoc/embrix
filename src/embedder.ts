/**
 * @fileoverview Core embedding functionality.
 * 
 * This module provides the Embedder class for generating text embeddings
 * using the loaded model pipelines.
 * 
 * Design Decision: Using a class-based API for better encapsulation
 * and to allow multiple embedder instances with different models.
 */

import { Tensor } from "@xenova/transformers";
import { loadModel } from "./loader";
import { EmbeddingModel, getModelConfig, ModelConfig } from "./models";

/**
 * Options for embedding generation.
 */
export interface EmbedOptions {
  /** Whether to normalize the output embedding (default: true) */
  normalize?: boolean;
  /** Pooling strategy to use (default: "mean") */
  pooling?: "mean" | "cls";
}

/**
 * Default embedding options.
 */
const DEFAULT_OPTIONS: Required<EmbedOptions> = {
  normalize: true,
  pooling: "mean"
};

/**
 * Result of an embedding operation with metadata.
 */
export interface EmbeddingResult {
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
export class Embedder {
  private readonly config: ModelConfig;

  /**
   * Create a new Embedder instance.
   * 
   * @param model - The embedding model to use
   * @throws Error if the model is not supported
   */
  constructor(private readonly model: EmbeddingModel) {
    this.config = getModelConfig(model);
  }

  /**
   * Get the dimension of embeddings produced by this model.
   * 
   * @returns The embedding dimension
   */
  get dimension(): number {
    return this.config.dimension;
  }

  /**
   * Get the model name.
   * 
   * @returns The human-readable model name
   */
  get modelName(): string {
    return this.config.name;
  }

  /**
   * Get the model enum value.
   * 
   * @returns The EmbeddingModel enum value
   */
  get modelType(): EmbeddingModel {
    return this.model;
  }

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
  async embed(text: string, options?: EmbedOptions): Promise<Float32Array> {
    // Validate input
    if (!text || typeof text !== "string") {
      throw new Error("Input text must be a non-empty string");
    }

    const opts = { ...DEFAULT_OPTIONS, ...options };
    const pipeline = await loadModel(this.model);

    try {
      // Generate embedding using the pipeline
      const output = await pipeline(text, {
        pooling: opts.pooling,
        normalize: opts.normalize
      });

      // Extract the embedding vector
      const vector = this.tensorToFloat32(output);

      // Validate dimension
      this.validateDimension(vector);

      return vector;
    } catch (error) {
      throw new Error(
        `Failed to generate embedding: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

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
  async embedBatch(texts: string[], options?: EmbedOptions): Promise<Float32Array[]> {
    // Validate input
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error("Input must be a non-empty array of strings");
    }

    for (let i = 0; i < texts.length; i++) {
      if (!texts[i] || typeof texts[i] !== "string") {
        throw new Error(`Invalid text at index ${i}: must be a non-empty string`);
      }
    }

    const opts = { ...DEFAULT_OPTIONS, ...options };
    const pipeline = await loadModel(this.model);

    try {
      // Generate embeddings for all texts
      const output = await pipeline(texts, {
        pooling: opts.pooling,
        normalize: opts.normalize
      });

      // Handle batch output - transformers.js returns different shapes
      // for single vs batch inputs
      const vectors = this.extractBatchVectors(output);

      // Validate all dimensions
      for (const vector of vectors) {
        this.validateDimension(vector);
      }

      return vectors;
    } catch (error) {
      throw new Error(
        `Failed to generate batch embeddings: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Generate an embedding with full result metadata.
   * 
   * @param text - The text to embed
   * @param options - Optional embedding configuration
   * @returns Promise resolving to embedding result with metadata
   */
  async embedWithMetadata(text: string, options?: EmbedOptions): Promise<EmbeddingResult> {
    const embedding = await this.embed(text, options);
    return {
      embedding,
      model: this.model,
      dimension: this.dimension
    };
  }

  /**
   * Check if the model for this embedder is loaded.
   * 
   * @returns true if the model is loaded and ready
   */
  isReady(): boolean {
    // Import here to avoid circular dependency
    const { isModelLoaded } = require("./loader");
    return isModelLoaded(this.model);
  }

  /**
   * Convert a Tensor to Float32Array.
   * 
   * Handles different tensor data types and shapes.
   */
  private tensorToFloat32(tensor: Tensor): Float32Array {
    // Get the raw data from the tensor
    const data = tensor.data;

    // If already Float32Array, return as-is
    if (data instanceof Float32Array) {
      return data;
    }

    // Convert other typed arrays to Float32Array
    return new Float32Array(data as ArrayLike<number>);
  }

  /**
   * Extract vectors from batch output tensor.
   * 
   * Handles the tensor shape from batch processing.
   */
  private extractBatchVectors(output: Tensor): Float32Array[] {
    const data = output.data;
    const dims = output.dims;

    // Convert data to Float32Array if needed
    const flatData = data instanceof Float32Array ? data : new Float32Array(data as ArrayLike<number>);

    // Handle different output shapes
    // For batch output, shape is typically [batch_size, hidden_dim]
    // or [batch_size, seq_len, hidden_dim] before pooling
    if (dims.length === 2) {
      // Shape: [batch_size, hidden_dim]
      const batchSize = dims[0]!;
      const hiddenDim = dims[1]!;
      const vectors: Float32Array[] = [];

      for (let i = 0; i < batchSize; i++) {
        const start = i * hiddenDim;
        const end = start + hiddenDim;
        vectors.push(flatData.slice(start, end));
      }

      return vectors;
    } else if (dims.length === 1) {
      // Single embedding returned (shape: [hidden_dim])
      return [this.tensorToFloat32(output)];
    } else if (dims.length === 3) {
      // Shape: [batch_size, seq_len, hidden_dim] - shouldn't happen with pooling
      // but handle gracefully
      const batchSize = dims[0]!;
      const seqLen = dims[1]!;
      const hiddenDim = dims[2]!;
      const vectors: Float32Array[] = [];

      // Take mean across sequence dimension for each batch
      for (let b = 0; b < batchSize; b++) {
        const vec = new Float32Array(hiddenDim);
        for (let s = 0; s < seqLen; s++) {
          for (let h = 0; h < hiddenDim; h++) {
            const idx = b * seqLen * hiddenDim + s * hiddenDim + h;
            vec[h]! += flatData[idx]! / seqLen;
          }
        }
        vectors.push(vec);
      }

      return vectors;
    }

    // Fallback: treat as single vector
    return [this.tensorToFloat32(output)];
  }

  /**
   * Validate that an embedding has the expected dimension.
   * 
   * @param vector - The embedding vector to validate
   * @throws Error if dimension doesn't match expected
   */
  private validateDimension(vector: Float32Array): void {
    const expected = this.config.dimension;

    if (vector.length !== expected) {
      throw new Error(
        `Embedding dimension mismatch: expected ${expected}, got ${vector.length}. ` +
        `This may indicate a model loading issue or incompatible model.`
      );
    }
  }
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
export function createEmbedder(model: EmbeddingModel): Embedder {
  return new Embedder(model);
}
