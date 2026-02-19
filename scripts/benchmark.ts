#!/usr/bin/env npx tsx
/**
 * @fileoverview CLI benchmark script for embrix.
 * 
 * This script runs comprehensive benchmarks for all supported embedding models
 * and outputs detailed performance metrics.
 * 
 * Usage:
 *   npm run benchmark
 *   npx tsx scripts/benchmark.ts
 *   npx tsx scripts/benchmark.ts --model minilm
 *   npx tsx scripts/benchmark.ts --batch-size 50
 */

import { runBenchmark, runAllBenchmarks, compareBenchmarks, formatBenchmarkResult } from "../src/benchmark";
import { EmbeddingModel, getSupportedModels, MODEL_CONFIG } from "../src/models";
import { clearModelCache } from "../src/loader";

/**
 * Parse command line arguments.
 */
function parseArgs(): { model?: EmbeddingModel; batchSize: number; compare: boolean } {
  const args = process.argv.slice(2);
  let model: EmbeddingModel | undefined;
  let batchSize = 100;
  let compare = true;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--model" || arg === "-m") {
      const modelArg = args[++i];
      if (modelArg === "minilm") {
        model = EmbeddingModel.MiniLM;
      } else if (modelArg === "bge") {
        model = EmbeddingModel.BGE;
      } else {
        console.error(`Unknown model: ${modelArg}`);
        console.error(`Valid models: minilm, bge`);
        process.exit(1);
      }
    } else if (arg === "--batch-size" || arg === "-b") {
      batchSize = parseInt(args[++i]!, 10);
      if (isNaN(batchSize) || batchSize < 1) {
        console.error("Batch size must be a positive integer");
        process.exit(1);
      }
    } else if (arg === "--no-compare") {
      compare = false;
    } else if (arg === "--help" || arg === "-h") {
      console.log(`
embrix Benchmark CLI

Usage:
  npm run benchmark [options]

Options:
  -m, --model <model>      Model to benchmark (minilm, bge)
                           If not specified, benchmarks all models
  -b, --batch-size <n>     Number of texts for batch benchmark (default: 100)
  --no-compare             Skip model comparison when benchmarking all models
  -h, --help               Show this help message

Examples:
  npm run benchmark
  npm run benchmark -- --model minilm
  npm run benchmark -- --batch-size 50
`);
      process.exit(0);
    }
  }

  return { model, batchSize, compare };
}

/**
 * Main entry point.
 */
async function main(): Promise<void> {
  console.log("\n╔════════════════════════════════════════════════════════════╗");
  console.log("║                 EMBRIX BENCHMARK SUITE                     ║");
  console.log("║         Local Text Embedding Performance Tests              ║");
  console.log("╚════════════════════════════════════════════════════════════╝\n");

  const { model, batchSize, compare } = parseArgs();

  // Display supported models
  console.log("Supported Models:");
  for (const m of getSupportedModels()) {
    const config = MODEL_CONFIG[m];
    console.log(`  - ${config.name} (${m}): ${config.dimension}D`);
  }
  console.log("");

  try {
    if (model) {
      // Benchmark single model
      console.log(`Benchmarking model: ${model}\n`);
      const result = await runBenchmark(model, { batchSize, verbose: true });
      console.log(formatBenchmarkResult(result));
    } else {
      // Benchmark all models
      console.log("Benchmarking all models...\n");
      const results = await runAllBenchmarks({ batchSize, verbose: true });

      // Print individual results
      for (const result of results) {
        console.log(formatBenchmarkResult(result));
      }

      // Compare models if more than one
      if (compare && results.length > 1) {
        console.log(compareBenchmarks(results[0]!, results[1]!));
      }
    }

    console.log("\n✅ Benchmark completed successfully!\n");
  } catch (error) {
    console.error("\n❌ Benchmark failed:");
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  } finally {
    // Clean up
    clearModelCache();
  }
}

// Run the benchmark
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
