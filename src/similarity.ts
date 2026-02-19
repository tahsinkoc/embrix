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
export function dotProduct(a: Float32Array, b: Float32Array): number {
  validateEqualLength(a, b);

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!;
  }

  return sum;
}

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
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  validateEqualLength(a, b);

  let dotSum = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i]!;
    const bVal = b[i]!;
    dotSum += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA * normB);

  if (magnitude === 0) {
    throw new Error("Cannot compute cosine similarity: zero magnitude vector");
  }

  return dotSum / magnitude;
}

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
export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  validateEqualLength(a, b);

  let sumSquared = 0;

  for (let i = 0; i < a.length; i++) {
    const diff = a[i]! - b[i]!;
    sumSquared += diff * diff;
  }

  return Math.sqrt(sumSquared);
}

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
export function euclideanDistanceSquared(a: Float32Array, b: Float32Array): number {
  validateEqualLength(a, b);

  let sumSquared = 0;

  for (let i = 0; i < a.length; i++) {
    const diff = a[i]! - b[i]!;
    sumSquared += diff * diff;
  }

  return sumSquared;
}

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
export function manhattanDistance(a: Float32Array, b: Float32Array): number {
  validateEqualLength(a, b);

  let sum = 0;

  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i]! - b[i]!);
  }

  return sum;
}

/**
 * Calculate the magnitude (L2 norm) of a vector.
 * 
 * @param vector - The input vector
 * @returns The magnitude
 */
export function magnitude(vector: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    const val = vector[i]!;
    sum += val * val;
  }
  return Math.sqrt(sum);
}

/**
 * Normalize a vector to unit length.
 * 
 * Returns a new vector with the same direction but magnitude 1.
 * 
 * @param vector - The input vector
 * @returns A new normalized vector
 * @throws Error if vector has zero magnitude
 */
export function normalize(vector: Float32Array): Float32Array {
  const mag = magnitude(vector);

  if (mag === 0) {
    throw new Error("Cannot normalize zero magnitude vector");
  }

  const result = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    result[i] = vector[i]! / mag;
  }

  return result;
}

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
export function findMostSimilar(
  query: Float32Array,
  candidates: Float32Array[]
): { index: number; similarity: number } {
  if (candidates.length === 0) {
    throw new Error("Candidates array cannot be empty");
  }

  let bestIndex = 0;
  let bestSimilarity = -Infinity;

  for (let i = 0; i < candidates.length; i++) {
    const similarity = cosineSimilarity(query, candidates[i]!);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestIndex = i;
    }
  }

  return { index: bestIndex, similarity: bestSimilarity };
}

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
export function findKMostSimilar(
  query: Float32Array,
  candidates: Float32Array[],
  k: number
): Array<{ index: number; similarity: number }> {
  if (candidates.length === 0) {
    throw new Error("Candidates array cannot be empty");
  }

  // Calculate all similarities
  const similarities = candidates.map((candidate, index) => ({
    index,
    similarity: cosineSimilarity(query, candidate)
  }));

  // Sort by similarity descending
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Return top k
  return similarities.slice(0, Math.min(k, similarities.length));
}

/**
 * Validate that two vectors have equal length.
 * 
 * @throws Error if vectors have different lengths
 */
function validateEqualLength(a: Float32Array, b: Float32Array): void {
  if (a.length !== b.length) {
    throw new Error(
      `Vector length mismatch: ${a.length} vs ${b.length}. ` +
      "All similarity functions require vectors of equal length."
    );
  }
}
