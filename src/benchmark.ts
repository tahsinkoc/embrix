/**
 * @fileoverview Benchmark utilities for measuring embedding performance.
 * 
 * This module provides tools to measure and analyze embedding generation
 * performance, including cold/warm start times and batch throughput.
 * 
 * Design Decision: Using process.hrtime.bigint() for nanosecond precision
 * timing without external dependencies.
 */

import { Embedder } from "./embedder";
import { EmbeddingModel } from "./models";

/**
 * Result of a single benchmark measurement.
 */
export interface BenchmarkResult {
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
export interface BenchmarkSuiteResult {
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
export interface BenchmarkOptions {
  /** Number of texts for batch benchmark (default: 100) */
  batchSize?: number;
  /** Text to use for single embedding benchmarks */
  sampleText?: string;
  /** Whether to log progress to console */
  verbose?: boolean;
}

/**
 * Default benchmark options.
 */
const DEFAULT_OPTIONS: Required<BenchmarkOptions> = {
  batchSize: 100,
  sampleText: "This is a sample text for benchmarking embedding generation performance.",
  verbose: true
};

/**
 * Generate sample texts for batch benchmarking.
 * 
 * @param count - Number of texts to generate
 * @returns Array of sample texts
 */
function generateSampleTexts(count: number): string[] {
  const templates = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can generate embeddings for text.",
    "Natural language processing enables computers to understand human language.",
    "Vector embeddings capture semantic meaning in dense numerical representations.",
    "Transformers have revolutionized the field of natural language processing."
  ];

  const texts: string[] = [];
  for (let i = 0; i < count; i++) {
    // Cycle through templates and add variation
    const template = templates[i % templates.length]!;
    texts.push(`${template} (sample ${i + 1})`);
  }

  return texts;
}

/**
 * Measure execution time of an async operation.
 * 
 * @param fn - Async function to measure
 * @returns Object with duration in milliseconds and result
 */
async function measureTime<T>(
  fn: () => Promise<T>
): Promise<{ durationMs: number; result: T }> {
  const start = process.hrtime.bigint();
  const result = await fn();
  const end = process.hrtime.bigint();

  // Convert nanoseconds to milliseconds
  const durationNs = Number(end - start);
  const durationMs = durationNs / 1_000_000;

  return { durationMs, result };
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
export async function runBenchmark(
  model: EmbeddingModel,
  options?: BenchmarkOptions
): Promise<BenchmarkSuiteResult> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const embedder = new Embedder(model);

  const suiteStart = process.hrtime.bigint();

  // Log start
  if (opts.verbose) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`Benchmark: ${embedder.modelName}`);
    console.log(`Model: ${model}`);
    console.log(`Dimension: ${embedder.dimension}`);
    console.log(`${"=".repeat(60)}\n`);
  }

  // Cold start benchmark (includes model loading)
  if (opts.verbose) {
    console.log("Running cold start benchmark...");
  }
  const coldStartResult = await measureTime(() =>
    embedder.embed(opts.sampleText)
  );
  const coldStart: BenchmarkResult = {
    operation: "Cold Start (first embed)",
    durationMs: coldStartResult.durationMs,
    embeddingCount: 1,
    avgTimePerEmbeddingMs: coldStartResult.durationMs,
    throughput: 1000 / coldStartResult.durationMs
  };

  if (opts.verbose) {
    console.log(`  Duration: ${coldStart.durationMs.toFixed(2)}ms`);
  }

  // Warm start benchmark (model already loaded)
  if (opts.verbose) {
    console.log("\nRunning warm start benchmark...");
  }
  const warmStartResult = await measureTime(() =>
    embedder.embed(opts.sampleText)
  );
  const warmStart: BenchmarkResult = {
    operation: "Warm Start (second embed)",
    durationMs: warmStartResult.durationMs,
    embeddingCount: 1,
    avgTimePerEmbeddingMs: warmStartResult.durationMs,
    throughput: 1000 / warmStartResult.durationMs
  };

  if (opts.verbose) {
    console.log(`  Duration: ${warmStart.durationMs.toFixed(2)}ms`);
    console.log(`  Speedup: ${(coldStart.durationMs / warmStart.durationMs).toFixed(2)}x faster than cold start`);
  }

  // Batch benchmark
  if (opts.verbose) {
    console.log(`\nRunning batch benchmark (${opts.batchSize} texts)...`);
  }
  const sampleTexts = generateSampleTexts(opts.batchSize);
  const batchResult = await measureTime(() =>
    embedder.embedBatch(sampleTexts)
  );
  const batchBenchmark: BenchmarkResult = {
    operation: `Batch Embed (${opts.batchSize} texts)`,
    durationMs: batchResult.durationMs,
    embeddingCount: opts.batchSize,
    avgTimePerEmbeddingMs: batchResult.durationMs / opts.batchSize,
    throughput: (opts.batchSize * 1000) / batchResult.durationMs
  };

  if (opts.verbose) {
    console.log(`  Total duration: ${batchBenchmark.durationMs.toFixed(2)}ms`);
    console.log(`  Avg per embedding: ${batchBenchmark.avgTimePerEmbeddingMs.toFixed(2)}ms`);
    console.log(`  Throughput: ${batchBenchmark.throughput.toFixed(2)} embeddings/sec`);
  }

  const suiteEnd = process.hrtime.bigint();
  const totalDurationMs = Number(suiteEnd - suiteStart) / 1_000_000;

  // Summary
  if (opts.verbose) {
    console.log(`\n${"-".repeat(60)}`);
    console.log("Summary:");
    console.log(`  Total benchmark time: ${totalDurationMs.toFixed(2)}ms`);
    console.log(`  Cold start overhead: ${(coldStart.durationMs - warmStart.durationMs).toFixed(2)}ms`);
    console.log(`${"-".repeat(60)}\n`);
  }

  return {
    model,
    modelName: embedder.modelName,
    dimension: embedder.dimension,
    coldStart,
    warmStart,
    batchBenchmark,
    totalDurationMs,
    timestamp: new Date().toISOString()
  };
}

/**
 * Run benchmarks for all supported models.
 * 
 * @param options - Benchmark configuration options
 * @returns Array of benchmark results for each model
 */
export async function runAllBenchmarks(
  options?: BenchmarkOptions
): Promise<BenchmarkSuiteResult[]> {
  const models = Object.values(EmbeddingModel);
  const results: BenchmarkSuiteResult[] = [];

  for (const model of models) {
    const result = await runBenchmark(model, options);
    results.push(result);
  }

  return results;
}

/**
 * Format benchmark results as a printable string.
 * 
 * @param result - Benchmark suite result
 * @returns Formatted string representation
 */
export function formatBenchmarkResult(result: BenchmarkSuiteResult): string {
  const lines: string[] = [
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

/**
 * Compare benchmark results between two models.
 * 
 * @param result1 - First model's benchmark results
 * @param result2 - Second model's benchmark results
 * @returns Comparison summary string
 */
export function compareBenchmarks(
  result1: BenchmarkSuiteResult,
  result2: BenchmarkSuiteResult
): string {
  const lines: string[] = [
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
