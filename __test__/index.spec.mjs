import test from 'ava';

import { computeCosineSimilarity } from '../index.js';

// Test case 1: vectors are the same
test('Cosine similarity between identical vectors should be 1', (t) => {
  const vector1 = [1, 2, 3];
  const vector2 = [1, 2, 3];
  const result = computeCosineSimilarity(vector1, vector2);
  t.is(result, 1);
});

// Test case 2: vectors are orthogonal
test('Cosine similarity between orthogonal vectors should be 0', (t) => {
  const vector1 = [1, 0, 0];
  const vector2 = [0, 1, 0];
  const result = computeCosineSimilarity(vector1, vector2);
  t.is(result, 0);
});

// Test case 3: vectors are opposite
test('Cosine similarity between opposite vectors should be -1', (t) => {
  const vector1 = [1, 2, 3];
  const vector2 = [-1, -2, -3];
  const result = computeCosineSimilarity(vector1, vector2);
  t.is(result, -1);
});

// Test case 4: vectors are random
test('Cosine similarity between random vectors should be between -1 and 1', (t) => {
  const vector1 = [0.5, 0.2, 0.9];
  const vector2 = [0.1, 0.6, 0.3];
  const result = computeCosineSimilarity(vector1, vector2);
  t.true(result >= -1 && result <= 1);
});

// Test case 5: one of the vectors is a zero vector
test('Cosine similarity with a zero vector should always be 0', (t) => {
  const vector1 = [0, 0, 0];
  const vector2 = [1, 2, 3];
  const result = computeCosineSimilarity(vector1, vector2);
  t.is(result, 0);
});
