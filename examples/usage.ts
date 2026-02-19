/**
 * @fileoverview Example usage of the embrix embedding library.
 * 
 * This file demonstrates the main features of the library:
 * - Creating embedders
 * - Generating single and batch embeddings
 * - Computing similarity between embeddings
 * - Finding similar documents
 * 
 * Run with: npx tsx examples/usage.ts
 */

import {
  Embedder,
  EmbeddingModel,
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  findMostSimilar,
  findKMostSimilar,
  preloadModel
} from "../src";

/**
 * Demonstrate basic embedding generation.
 */
async function basicEmbeddingExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Basic Embedding Example");
  console.log("=".repeat(60) + "\n");

  // Create an embedder with MiniLM model
  const embedder = new Embedder(EmbeddingModel.MiniLM);

  console.log(`Model: ${embedder.modelName}`);
  console.log(`Dimension: ${embedder.dimension}`);

  // Generate a single embedding
  const text = "Hello, world! This is a test of the embedding system.";
  console.log(`\nEmbedding text: "${text}"`);

  const embedding = await embedder.embed(text);

  console.log(`Embedding length: ${embedding.length}`);
  console.log(`First 5 values: [${Array.from(embedding.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}]`);
}

/**
 * Demonstrate batch embedding generation.
 */
async function batchEmbeddingExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Batch Embedding Example");
  console.log("=".repeat(60) + "\n");

  const embedder = new Embedder(EmbeddingModel.MiniLM);

  // Generate embeddings for multiple texts
  const texts = [
    "The cat sat on the mat.",
    "A feline was resting on a rug.",
    "The stock market crashed yesterday.",
    "I love programming in TypeScript.",
    "Machine learning is transforming technology."
  ];

  console.log("Embedding texts:");
  texts.forEach((t, i) => console.log(`  ${i + 1}. "${t}"`));

  const embeddings = await embedder.embedBatch(texts);

  console.log(`\nGenerated ${embeddings.length} embeddings`);
  console.log(`Each embedding has ${embeddings[0]!.length} dimensions`);
}

/**
 * Demonstrate similarity computation.
 */
async function similarityExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Similarity Example");
  console.log("=".repeat(60) + "\n");

  const embedder = new Embedder(EmbeddingModel.MiniLM);

  // Generate embeddings for similar and dissimilar texts
  const cat1 = await embedder.embed("The cat sat on the mat.");
  const cat2 = await embedder.embed("A feline was resting on a rug.");
  const market = await embedder.embed("The stock market crashed yesterday.");

  // Compute similarities
  console.log("Comparing embeddings:\n");

  const cat1Cat2Similarity = cosineSimilarity(cat1, cat2);
  console.log(`"The cat sat on the mat." vs "A feline was resting on a rug."`);
  console.log(`  Cosine Similarity: ${cat1Cat2Similarity.toFixed(4)}`);
  console.log(`  (High similarity - similar meaning)\n`);

  const cat1MarketSimilarity = cosineSimilarity(cat1, market);
  console.log(`"The cat sat on the mat." vs "The stock market crashed yesterday."`);
  console.log(`  Cosine Similarity: ${cat1MarketSimilarity.toFixed(4)}`);
  console.log(`  (Low similarity - different topics)\n`);

  // Other distance metrics
  console.log("Other distance metrics:");
  console.log(`  Dot Product (cat1 vs cat2): ${dotProduct(cat1, cat2).toFixed(4)}`);
  console.log(`  Euclidean Distance (cat1 vs cat2): ${euclideanDistance(cat1, cat2).toFixed(4)}`);
}

/**
 * Demonstrate semantic search.
 */
async function semanticSearchExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Semantic Search Example");
  console.log("=".repeat(60) + "\n");

  const embedder = new Embedder(EmbeddingModel.MiniLM);

  // Document corpus
  const documents = [
    "JavaScript is a programming language for web development.",
    "Python is popular for data science and machine learning.",
    "TypeScript adds static typing to JavaScript.",
    "Rust is a systems programming language focused on safety.",
    "Go is designed for concurrent programming and cloud services."
  ];

  // Embed all documents
  console.log("Document corpus:");
  documents.forEach((d, i) => console.log(`  [${i}] ${d}`));

  const docEmbeddings = await embedder.embedBatch(documents);

  // Search query
  const query = "What language is good for AI and ML?";
  console.log(`\nSearch query: "${query}"`);

  const queryEmbedding = await embedder.embed(query);

  // Find most similar document
  const best = findMostSimilar(queryEmbedding, docEmbeddings);
  console.log(`\nBest match:`);
  console.log(`  Document [${best.index}]: "${documents[best.index]}"`);
  console.log(`  Similarity: ${best.similarity.toFixed(4)}`);

  // Find top 3 similar documents
  const top3 = findKMostSimilar(queryEmbedding, docEmbeddings, 3);
  console.log(`\nTop 3 matches:`);
  top3.forEach((result, rank) => {
    console.log(`  ${rank + 1}. [${result.index}] "${documents[result.index]}"`);
    console.log(`     Similarity: ${result.similarity.toFixed(4)}`);
  });
}

/**
 * Demonstrate using different models.
 */
async function multipleModelsExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Multiple Models Example");
  console.log("=".repeat(60) + "\n");

  const text = "This is a test sentence for comparing models.";

  // MiniLM model
  const miniLMEmbedder = new Embedder(EmbeddingModel.MiniLM);
  const miniLMEmbedding = await miniLMEmbedder.embed(text);

  console.log(`MiniLM (${miniLMEmbedder.modelName}):`);
  console.log(`  Dimension: ${miniLMEmbedder.dimension}`);
  console.log(`  First 3 values: [${Array.from(miniLMEmbedding.slice(0, 3)).map(v => v.toFixed(4)).join(", ")}]`);

  // BGE model
  const bgeEmbedder = new Embedder(EmbeddingModel.BGE);
  const bgeEmbedding = await bgeEmbedder.embed(text);

  console.log(`\nBGE (${bgeEmbedder.modelName}):`);
  console.log(`  Dimension: ${bgeEmbedder.dimension}`);
  console.log(`  First 3 values: [${Array.from(bgeEmbedding.slice(0, 3)).map(v => v.toFixed(4)).join(", ")}]`);

  // Note: Embeddings from different models are not directly comparable
  console.log("\nNote: Embeddings from different models use different vector spaces");
  console.log("and are not directly comparable via similarity metrics.");
}

/**
 * Demonstrate preloading models.
 */
async function preloadingExample(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("Model Preloading Example");
  console.log("=".repeat(60) + "\n");

  console.log("Preloading MiniLM model...");
  const startPreload = Date.now();
  await preloadModel(EmbeddingModel.MiniLM);
  console.log(`Model loaded in ${Date.now() - startPreload}ms`);

  // Now embedding will be fast
  const embedder = new Embedder(EmbeddingModel.MiniLM);
  const startEmbed = Date.now();
  await embedder.embed("Quick embedding after preload!");
  console.log(`Embedding generated in ${Date.now() - startEmbed}ms (warm start)`);
}

/**
 * Main entry point.
 */
async function main(): Promise<void> {
  console.log("\n╔════════════════════════════════════════════════════════════╗");
  console.log("║              EMBRIX - Embedding Library Examples           ║");
  console.log("╚════════════════════════════════════════════════════════════╝");

  try {
    await basicEmbeddingExample();
    await batchEmbeddingExample();
    await similarityExample();
    await semanticSearchExample();
    await multipleModelsExample();
    await preloadingExample();

    console.log("\n" + "=".repeat(60));
    console.log("✅ All examples completed successfully!");
    console.log("=".repeat(60) + "\n");
  } catch (error) {
    console.error("\n❌ Example failed:");
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

// Run the examples
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
