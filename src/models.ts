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
export enum EmbeddingModel {
  /** Xenova/all-MiniLM-L6-v2 - Fast and efficient, 384 dimensions */
  MiniLM = "minilm",
  /** Xenova/bge-small-en-v1.5 - High quality English embeddings, 384 dimensions */
  BGE = "bge"
}

/**
 * Configuration for an embedding model.
 */
export interface ModelConfig {
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
export const MODEL_CONFIG: Readonly<Record<EmbeddingModel, ModelConfig>> = {
  [EmbeddingModel.MiniLM]: {
    hfPath: "Xenova/all-MiniLM-L6-v2",
    dimension: 384,
    name: "all-MiniLM-L6-v2",
    description: "Fast and efficient sentence transformer model",
    maxLength: 256
  },
  [EmbeddingModel.BGE]: {
    hfPath: "Xenova/bge-small-en-v1.5",
    dimension: 384,
    name: "bge-small-en-v1.5",
    description: "High quality English embeddings from BAAI",
    maxLength: 512
  }
} as const;

/**
 * Type guard to check if a string is a valid EmbeddingModel.
 */
export function isValidModel(model: string): model is EmbeddingModel {
  return Object.values(EmbeddingModel).includes(model as EmbeddingModel);
}

/**
 * Get model configuration with validation.
 * Throws an error if the model is not found.
 */
export function getModelConfig(model: EmbeddingModel): ModelConfig {
  const config = MODEL_CONFIG[model];
  if (!config) {
    throw new Error(`Unknown model: ${model}. Valid models: ${Object.values(EmbeddingModel).join(", ")}`);
  }
  return config;
}

/**
 * Get all supported model names.
 */
export function getSupportedModels(): readonly EmbeddingModel[] {
  return Object.values(EmbeddingModel);
}
