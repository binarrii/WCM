import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/views/video-review.css', import.meta.url), 'utf8');

test('review results list stretches through the remaining panel height', () => {
  assert.match(css, /\.review-events\s*\{[^}]*\bflex:\s*1\b[^}]*\}/);
});
