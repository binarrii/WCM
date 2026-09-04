import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/views/review-tasks.css', import.meta.url), 'utf8');

test('review task table fills the viewport while keeping a bottom margin', () => {
  assert.match(css, /\.review-tasks\s*\{[^}]*min-height:\s*calc\(100dvh\s*-\s*108px\)[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.task-table-card\s*\{[^}]*flex:\s*1[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.task-table-scroll\s*\{[^}]*flex:\s*1[^}]*\}/);
});
