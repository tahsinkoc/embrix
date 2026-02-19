/**
 * @fileoverview Lazy singleton model loader for embedding pipelines.
 * 
 * This module implements a lazy loading pattern with in-memory caching
 * to ensure models are loaded only once during the application lifecycle.
 * 
 * Design Decision: Using Map cache instead of class-based singleton
 * for simpler API and better tree-shaking support.
 */

import { pipeline, FeatureExtractionPipeline, env } from "@xenova/transformers";
import { EmbeddingModel, MODEL_CONFIG, ModelConfig } from "./models";

/**
 * Configure transformers.js environment.
 * 
 * Design Decision: Disable local model check to allow downloading
 * from HuggingFace Hub. Set cache directory for model persistence.
 */
env.allowLocalModels = false;

/**
 * In-memory cache for loaded model pipelines.
 * 
 * Design Decision: Using Map for O(1) lookup and clear semantics.
 * The cache is module-scoped to ensure true singleton behavior.
 */
const modelCache = new Map<EmbeddingModel, FeatureExtractionPipeline>();

/**
 * Track loading state to prevent concurrent loads of the same model.
 */
const loadingPromises = new Map<EmbeddingModel, Promise<FeatureExtractionPipeline>>();

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
export async function loadModel(model: EmbeddingModel): Promise<FeatureExtractionPipeline> {
  // Return cached model if available
  if (modelCache.has(model)) {
    return modelCache.get(model)!;
  }

  // Return existing loading promise if model is being loaded
  if (loadingPromises.has(model)) {
    return loadingPromises.get(model)!;
  }

  // Start loading the model
  const config: ModelConfig = MODEL_CONFIG[model];
  
  const loadingPromise = pipeline("feature-extraction", config.hfPath, {
    // Use default quantized model for smaller download
    quantized: true,
  })
    .then((embedder) => {
      // Cache the loaded model
      modelCache.set(model, embedder);
      // Clear the loading promise
      loadingPromises.delete(model);
      return embedder;
    })
    .catch((error) => {
      // Clear the loading promise on failure
      loadingPromises.delete(model);
      throw new Error(
        `Failed to load model ${config.name} (${config.hfPath}): ${error instanceof Error ? error.message : String(error)}`
      );
    });

  // Store the loading promise
  loadingPromises.set(model, loadingPromise);

  return loadingPromise;
}

/**
 * Check if a model is already loaded and cached.
 * 
 * @param model - The embedding model to check
 * @returns true if model is cached and ready to use
 */
export function isModelLoaded(model: EmbeddingModel): boolean {
  return modelCache.has(model);
}

/**
 * Clear the model cache.
 * 
 * Useful for freeing memory or forcing model reload.
 * Warning: This will require re-downloading models on next use.
 */
export function clearModelCache(): void {
  modelCache.clear();
  loadingPromises.clear();
}

/**
 * Get list of currently loaded models.
 */
export function getLoadedModels(): EmbeddingModel[] {
  return Array.from(modelCache.keys());
}

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
export async function preloadModel(model: EmbeddingModel): Promise<void> {
  await loadModel(model);
}

/**
 * Preload all supported models.
 * 
 * @returns Promise that resolves when all models are loaded
 */
export async function preloadAllModels(): Promise<void> {
  const models = Object.values(EmbeddingModel);
  await Promise.all(models.map((model) => loadModel(model)));
}
