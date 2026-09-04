import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/views/review-tasks.css', import.meta.url), 'utf8');
const component = readFileSync(new URL('../src/views/ReviewTasks.vue', import.meta.url), 'utf8');
const service = readFileSync(new URL('../src/services/reviewTaskService.js', import.meta.url), 'utf8');

test('review task table fills the viewport while keeping a bottom margin', () => {
  assert.match(css, /\.review-tasks\s*\{[^}]*min-height:\s*calc\(100dvh\s*-\s*108px\)[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.task-table-card\s*\{[^}]*flex:\s*1[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.task-table-scroll\s*\{[^}]*flex:\s*1[^}]*\}/);
});

test('action column remains a table cell so row separators span the full width', () => {
  const actionRule = css.match(/\.task-actions\s*\{([^}]*)\}/)?.[1] ?? '';

  assert.doesNotMatch(actionRule, /\bdisplay:\s*flex\b/);
  assert.match(actionRule, /text-align:\s*right/);
  assert.match(css, /\.task-actions button\s*\{[^}]*display:\s*inline-grid/);
});

test('completed task results support single JSON and selected ZIP downloads', () => {
  assert.match(component, /title="task\.status === 'completed' \? '下载分析结果'/);
  assert.match(component, /task\.status !== 'completed'/);
  assert.match(component, /批量下载 \(\$\{downloadableSelectedIds\.length\}\)/);
  assert.match(service, /responseType:\s*'blob'/);
  assert.match(service, /review_tasks\/results\/download/);
});
