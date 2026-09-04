import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const dialog = readFileSync(new URL('../src/components/ConfirmDialog.vue', import.meta.url), 'utf8');
const taskView = readFileSync(new URL('../src/views/ReviewTasks.vue', import.meta.url), 'utf8');
const faceView = readFileSync(new URL('../src/views/FaceDashboard.vue', import.meta.url), 'utf8');

test('reusable confirmation dialog follows accessibility and theme contracts', () => {
  assert.match(dialog, /<Teleport to="body">/);
  assert.match(dialog, /role="alertdialog"/);
  assert.match(dialog, /aria-modal="true"/);
  assert.match(dialog, /var\(--modal-card-bg\)/);
  assert.match(dialog, /var\(--modal-overlay-bg\)/);
  assert.match(dialog, /emit\('confirm'\)/);
  assert.match(dialog, /emit\('cancel'\)/);
});

test('destructive actions use the reusable dialog instead of browser confirmation', () => {
  assert.doesNotMatch(`${taskView}\n${faceView}`, /(?:window\.)?confirm\s*\(/);
  assert.match(taskView, /import ConfirmDialog/);
  assert.match(faceView, /import ConfirmDialog/);
  assert.match(taskView, /<ConfirmDialog/);
  assert.match(faceView, /<ConfirmDialog/);
});
