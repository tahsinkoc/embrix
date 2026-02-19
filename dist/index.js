var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/models.ts
function isValidModel(model) {
  return Object.values(EmbeddingModel).includes(model);
}
function getModelConfig(model) {
  const config = MODEL_CONFIG[model];
  if (!config) {
    throw new Error(`Unknown model: ${model}. Valid models: ${Object.values(EmbeddingModel).join(", ")}`);
  }
  return config;
}
function getSupportedModels() {
  return Object.values(EmbeddingModel);
}
var EmbeddingModel, MODEL_CONFIG;
var init_models = __esm({
  "src/models.ts"() {
    "use strict";
    EmbeddingModel = /* @__PURE__ */ ((EmbeddingModel3) => {
      EmbeddingModel3["MiniLM"] = "minilm";
      EmbeddingModel3["BGE"] = "bge";
      return EmbeddingModel3;
    })(EmbeddingModel || {});
    MODEL_CONFIG = {
      ["minilm" /* MiniLM */]: {
        hfPath: "Xenova/all-MiniLM-L6-v2",
        dimension: 384,
        name: "all-MiniLM-L6-v2",
        description: "Fast and efficient sentence transformer model",
        maxLength: 256
      },
      ["bge" /* BGE */]: {
        hfPath: "Xenova/bge-small-en-v1.5",
        dimension: 384,
        name: "bge-small-en-v1.5",
        description: "High quality English embeddings from BAAI",
        maxLength: 512
      }
    };
  }
});

// src/loader.ts
var loader_exports = {};
__export(loader_exports, {
  clearModelCache: () => clearModelCache,
  getLoadedModels: () => getLoadedModels,
  isModelLoaded: () => isModelLoaded,
  loadModel: () => loadModel,
  preloadAllModels: () => preloadAllModels,
  preloadModel: () => preloadModel
});
import { pipeline, env } from "@xenova/transformers";
async function loadModel(model) {
  if (modelCache.has(model)) {
    return modelCache.get(model);
  }
  if (loadingPromises.has(model)) {
    return loadingPromises.get(model);
  }
  const config = MODEL_CONFIG[model];
  const loadingPromise = pipeline("feature-extraction", config.hfPath, {
    // Use default quantized model for smaller download
    quantized: true
  }).then((embedder) => {
    modelCache.set(model, embedder);
    loadingPromises.delete(model);
    return embedder;
  }).catch((error) => {
    loadingPromises.delete(model);
    throw new Error(
      `Failed to load model ${config.name} (${config.hfPath}): ${error instanceof Error ? error.message : String(error)}`
    );
  });
  loadingPromises.set(model, loadingPromise);
  return loadingPromise;
}
function isModelLoaded(model) {
  return modelCache.has(model);
}
function clearModelCache() {
  modelCache.clear();
  loadingPromises.clear();
}
function getLoadedModels() {
  return Array.from(modelCache.keys());
}
async function preloadModel(model) {
  await loadModel(model);
}
async function preloadAllModels() {
  const models = Object.values(EmbeddingModel);
  await Promise.all(models.map((model) => loadModel(model)));
}
var modelCache, loadingPromises;
var init_loader = __esm({
  "src/loader.ts"() {
    "use strict";
    init_models();
    env.allowLocalModels = false;
    modelCache = /* @__PURE__ */ new Map();
    loadingPromises = /* @__PURE__ */ new Map();
  }
});

// src/index.ts
init_models();
init_loader();

// src/embedder.ts
init_loader();
init_models();
var DEFAULT_OPTIONS = {
  normalize: true,
  pooling: "mean"
};
var Embedder = class {
  /**
   * Create a new Embedder instance.
   * 
   * @param model - The embedding model to use
   * @throws Error if the model is not supported
   */
  constructor(model) {
    this.model = model;
    this.config = getModelConfig(model);
  }
  /**
   * Get the dimension of embeddings produced by this model.
   * 
   * @returns The embedding dimension
   */
  get dimension() {
    return this.config.dimension;
  }
  /**
   * Get the model name.
   * 
   * @returns The human-readable model name
   */
  get modelName() {
    return this.config.name;
  }
  /**
   * Get the model enum value.
   * 
   * @returns The EmbeddingModel enum value
   */
  get modelType() {
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
  async embed(text, options) {
    if (!text || typeof text !== "string") {
      throw new Error("Input text must be a non-empty string");
    }
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const pipeline2 = await loadModel(this.model);
    try {
      const output = await pipeline2(text, {
        pooling: opts.pooling,
        normalize: opts.normalize
      });
      const vector = this.tensorToFloat32(output);
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
  async embedBatch(texts, options) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error("Input must be a non-empty array of strings");
    }
    for (let i = 0; i < texts.length; i++) {
      if (!texts[i] || typeof texts[i] !== "string") {
        throw new Error(`Invalid text at index ${i}: must be a non-empty string`);
      }
    }
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const pipeline2 = await loadModel(this.model);
    try {
      const output = await pipeline2(texts, {
        pooling: opts.pooling,
        normalize: opts.normalize
      });
      const vectors = this.extractBatchVectors(output);
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
  async embedWithMetadata(text, options) {
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
  isReady() {
    const { isModelLoaded: isModelLoaded2 } = (init_loader(), __toCommonJS(loader_exports));
    return isModelLoaded2(this.model);
  }
  /**
   * Convert a Tensor to Float32Array.
   * 
   * Handles different tensor data types and shapes.
   */
  tensorToFloat32(tensor) {
    const data = tensor.data;
    if (data instanceof Float32Array) {
      return data;
    }
    return new Float32Array(data);
  }
  /**
   * Extract vectors from batch output tensor.
   * 
   * Handles the tensor shape from batch processing.
   */
  extractBatchVectors(output) {
    const data = output.data;
    const dims = output.dims;
    const flatData = data instanceof Float32Array ? data : new Float32Array(data);
    if (dims.length === 2) {
      const batchSize = dims[0];
      const hiddenDim = dims[1];
      const vectors = [];
      for (let i = 0; i < batchSize; i++) {
        const start = i * hiddenDim;
        const end = start + hiddenDim;
        vectors.push(flatData.slice(start, end));
      }
      return vectors;
    } else if (dims.length === 1) {
      return [this.tensorToFloat32(output)];
    } else if (dims.length === 3) {
      const batchSize = dims[0];
      const seqLen = dims[1];
      const hiddenDim = dims[2];
      const vectors = [];
      for (let b = 0; b < batchSize; b++) {
        const vec = new Float32Array(hiddenDim);
        for (let s = 0; s < seqLen; s++) {
          for (let h = 0; h < hiddenDim; h++) {
            const idx = b * seqLen * hiddenDim + s * hiddenDim + h;
            vec[h] += flatData[idx] / seqLen;
          }
        }
        vectors.push(vec);
      }
      return vectors;
    }
    return [this.tensorToFloat32(output)];
  }
  /**
   * Validate that an embedding has the expected dimension.
   * 
   * @param vector - The embedding vector to validate
   * @throws Error if dimension doesn't match expected
   */
  validateDimension(vector) {
    const expected = this.config.dimension;
    if (vector.length !== expected) {
      throw new Error(
        `Embedding dimension mismatch: expected ${expected}, got ${vector.length}. This may indicate a model loading issue or incompatible model.`
      );
    }
  }
};
function createEmbedder(model) {
  return new Embedder(model);
}

// src/similarity.ts
function dotProduct(a, b) {
  validateEqualLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}
function cosineSimilarity(a, b) {
  validateEqualLength(a, b);
  let dotSum = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const aVal = a[i];
    const bVal = b[i];
    dotSum += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }
  const magnitude2 = Math.sqrt(normA * normB);
  if (magnitude2 === 0) {
    throw new Error("Cannot compute cosine similarity: zero magnitude vector");
  }
  return dotSum / magnitude2;
}
function euclideanDistance(a, b) {
  validateEqualLength(a, b);
  let sumSquared = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSquared += diff * diff;
  }
  return Math.sqrt(sumSquared);
}
function euclideanDistanceSquared(a, b) {
  validateEqualLength(a, b);
  let sumSquared = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sumSquared += diff * diff;
  }
  return sumSquared;
}
function manhattanDistance(a, b) {
  validateEqualLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}
function magnitude(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    const val = vector[i];
    sum += val * val;
  }
  return Math.sqrt(sum);
}
function normalize(vector) {
  const mag = magnitude(vector);
  if (mag === 0) {
    throw new Error("Cannot normalize zero magnitude vector");
  }
  const result = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    result[i] = vector[i] / mag;
  }
  return result;
}
function findMostSimilar(query, candidates) {
  if (candidates.length === 0) {
    throw new Error("Candidates array cannot be empty");
  }
  let bestIndex = 0;
  let bestSimilarity = -Infinity;
  for (let i = 0; i < candidates.length; i++) {
    const similarity = cosineSimilarity(query, candidates[i]);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestIndex = i;
    }
  }
  return { index: bestIndex, similarity: bestSimilarity };
}
function findKMostSimilar(query, candidates, k) {
  if (candidates.length === 0) {
    throw new Error("Candidates array cannot be empty");
  }
  const similarities = candidates.map((candidate, index) => ({
    index,
    similarity: cosineSimilarity(query, candidate)
  }));
  similarities.sort((a, b) => b.similarity - a.similarity);
  return similarities.slice(0, Math.min(k, similarities.length));
}
function validateEqualLength(a, b) {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length}. All similarity functions require vectors of equal length.`
    );
  }
}

// src/benchmark.ts
init_models();
var DEFAULT_OPTIONS2 = {
  batchSize: 100,
  sampleText: "This is a sample text for benchmarking embedding generation performance.",
  verbose: true
};
function generateSampleTexts(count) {
  const templates = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can generate embeddings for text.",
    "Natural language processing enables computers to understand human language.",
    "Vector embeddings capture semantic meaning in dense numerical representations.",
    "Transformers have revolutionized the field of natural language processing."
  ];
  const texts = [];
  for (let i = 0; i < count; i++) {
    const template = templates[i % templates.length];
    texts.push(`${template} (sample ${i + 1})`);
  }
  return texts;
}
async function measureTime(fn) {
  const start = process.hrtime.bigint();
  const result = await fn();
  const end = process.hrtime.bigint();
  const durationNs = Number(end - start);
  const durationMs = durationNs / 1e6;
  return { durationMs, result };
}
async function runBenchmark(model, options) {
  const opts = { ...DEFAULT_OPTIONS2, ...options };
  const embedder = new Embedder(model);
  const suiteStart = process.hrtime.bigint();
  if (opts.verbose) {
    console.log(`
${"=".repeat(60)}`);
    console.log(`Benchmark: ${embedder.modelName}`);
    console.log(`Model: ${model}`);
    console.log(`Dimension: ${embedder.dimension}`);
    console.log(`${"=".repeat(60)}
`);
  }
  if (opts.verbose) {
    console.log("Running cold start benchmark...");
  }
  const coldStartResult = await measureTime(
    () => embedder.embed(opts.sampleText)
  );
  const coldStart = {
    operation: "Cold Start (first embed)",
    durationMs: coldStartResult.durationMs,
    embeddingCount: 1,
    avgTimePerEmbeddingMs: coldStartResult.durationMs,
    throughput: 1e3 / coldStartResult.durationMs
  };
  if (opts.verbose) {
    console.log(`  Duration: ${coldStart.durationMs.toFixed(2)}ms`);
  }
  if (opts.verbose) {
    console.log("\nRunning warm start benchmark...");
  }
  const warmStartResult = await measureTime(
    () => embedder.embed(opts.sampleText)
  );
  const warmStart = {
    operation: "Warm Start (second embed)",
    durationMs: warmStartResult.durationMs,
    embeddingCount: 1,
    avgTimePerEmbeddingMs: warmStartResult.durationMs,
    throughput: 1e3 / warmStartResult.durationMs
  };
  if (opts.verbose) {
    console.log(`  Duration: ${warmStart.durationMs.toFixed(2)}ms`);
    console.log(`  Speedup: ${(coldStart.durationMs / warmStart.durationMs).toFixed(2)}x faster than cold start`);
  }
  if (opts.verbose) {
    console.log(`
Running batch benchmark (${opts.batchSize} texts)...`);
  }
  const sampleTexts = generateSampleTexts(opts.batchSize);
  const batchResult = await measureTime(
    () => embedder.embedBatch(sampleTexts)
  );
  const batchBenchmark = {
    operation: `Batch Embed (${opts.batchSize} texts)`,
    durationMs: batchResult.durationMs,
    embeddingCount: opts.batchSize,
    avgTimePerEmbeddingMs: batchResult.durationMs / opts.batchSize,
    throughput: opts.batchSize * 1e3 / batchResult.durationMs
  };
  if (opts.verbose) {
    console.log(`  Total duration: ${batchBenchmark.durationMs.toFixed(2)}ms`);
    console.log(`  Avg per embedding: ${batchBenchmark.avgTimePerEmbeddingMs.toFixed(2)}ms`);
    console.log(`  Throughput: ${batchBenchmark.throughput.toFixed(2)} embeddings/sec`);
  }
  const suiteEnd = process.hrtime.bigint();
  const totalDurationMs = Number(suiteEnd - suiteStart) / 1e6;
  if (opts.verbose) {
    console.log(`
${"-".repeat(60)}`);
    console.log("Summary:");
    console.log(`  Total benchmark time: ${totalDurationMs.toFixed(2)}ms`);
    console.log(`  Cold start overhead: ${(coldStart.durationMs - warmStart.durationMs).toFixed(2)}ms`);
    console.log(`${"-".repeat(60)}
`);
  }
  return {
    model,
    modelName: embedder.modelName,
    dimension: embedder.dimension,
    coldStart,
    warmStart,
    batchBenchmark,
    totalDurationMs,
    timestamp: (/* @__PURE__ */ new Date()).toISOString()
  };
}
async function runAllBenchmarks(options) {
  const models = Object.values(EmbeddingModel);
  const results = [];
  for (const model of models) {
    const result = await runBenchmark(model, options);
    results.push(result);
  }
  return results;
}
function formatBenchmarkResult(result) {
  const lines = [
    "=".repeat(60),
    `Benchmark Results: ${result.modelName}`,
    "=".repeat(60),
    "",
    `Model: ${result.model}`,
    `Dimension: ${result.dimension}`,
    `Timestamp: ${result.timestamp}`,
    "",
    "-".repeat(60),
    "Performance Metrics:",
    "-".repeat(60),
    "",
    `Cold Start (first embed):`,
    `  Duration: ${result.coldStart.durationMs.toFixed(2)}ms`,
    `  Throughput: ${result.coldStart.throughput.toFixed(2)} embeddings/sec`,
    "",
    `Warm Start (subsequent embed):`,
    `  Duration: ${result.warmStart.durationMs.toFixed(2)}ms`,
    `  Throughput: ${result.warmStart.throughput.toFixed(2)} embeddings/sec`,
    "",
    `Batch Embed (${result.batchBenchmark.embeddingCount} texts):`,
    `  Total Duration: ${result.batchBenchmark.durationMs.toFixed(2)}ms`,
    `  Avg per Embedding: ${result.batchBenchmark.avgTimePerEmbeddingMs.toFixed(2)}ms`,
    `  Throughput: ${result.batchBenchmark.throughput.toFixed(2)} embeddings/sec`,
    "",
    "-".repeat(60),
    `Total Benchmark Time: ${result.totalDurationMs.toFixed(2)}ms`,
    "=".repeat(60)
  ];
  return lines.join("\n");
}
function compareBenchmarks(result1, result2) {
  const lines = [
    "\n" + "=".repeat(60),
    "Benchmark Comparison",
    "=".repeat(60),
    "",
    `Model 1: ${result1.modelName}`,
    `Model 2: ${result2.modelName}`,
    "",
    "-".repeat(60),
    "Cold Start Comparison:",
    "-".repeat(60),
    `  ${result1.modelName}: ${result1.coldStart.durationMs.toFixed(2)}ms`,
    `  ${result2.modelName}: ${result2.coldStart.durationMs.toFixed(2)}ms`,
    `  Difference: ${Math.abs(result1.coldStart.durationMs - result2.coldStart.durationMs).toFixed(2)}ms`,
    "",
    "-".repeat(60),
    "Warm Start Comparison:",
    "-".repeat(60),
    `  ${result1.modelName}: ${result1.warmStart.durationMs.toFixed(2)}ms`,
    `  ${result2.modelName}: ${result2.warmStart.durationMs.toFixed(2)}ms`,
    `  Difference: ${Math.abs(result1.warmStart.durationMs - result2.warmStart.durationMs).toFixed(2)}ms`,
    "",
    "-".repeat(60),
    "Batch Throughput Comparison:",
    "-".repeat(60),
    `  ${result1.modelName}: ${result1.batchBenchmark.throughput.toFixed(2)} embeddings/sec`,
    `  ${result2.modelName}: ${result2.batchBenchmark.throughput.toFixed(2)} embeddings/sec`,
    `  Difference: ${Math.abs(result1.batchBenchmark.throughput - result2.batchBenchmark.throughput).toFixed(2)} embeddings/sec`,
    "",
    "=".repeat(60)
  ];
  return lines.join("\n");
}
export {
  Embedder,
  EmbeddingModel,
  MODEL_CONFIG,
  clearModelCache,
  compareBenchmarks,
  cosineSimilarity,
  createEmbedder,
  dotProduct,
  euclideanDistance,
  euclideanDistanceSquared,
  findKMostSimilar,
  findMostSimilar,
  formatBenchmarkResult,
  getLoadedModels,
  getModelConfig,
  getSupportedModels,
  isModelLoaded,
  isValidModel,
  loadModel,
  magnitude,
  manhattanDistance,
  normalize,
  preloadAllModels,
  preloadModel,
  runAllBenchmarks,
  runBenchmark
};
