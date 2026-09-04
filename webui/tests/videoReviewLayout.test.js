import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/views/video-review.css', import.meta.url), 'utf8');
const component = readFileSync(new URL('../src/views/VideoReview.vue', import.meta.url), 'utf8');

test('review results list stretches through the remaining panel height', () => {
  assert.match(css, /\.review-workspace\s*\{[^}]*align-items:\s*start[^}]*\}/);
  assert.match(css, /\.review-events-panel\s*\{[^}]*height:\s*var\(--review-player-height\)[^}]*overflow:\s*hidden[^}]*\}/);
  assert.match(css, /\.review-events\s*\{[^}]*\bflex:\s*1\b[^}]*\}/);
  assert.match(component, /panelResizeObserver\s*=\s*new ResizeObserver\(updatePlayerPanelHeight\)/);
});

test('parameter panel can collapse and completed results collapse it automatically', () => {
  assert.match(component, /setupExpanded\s*=\s*ref\(true\)/);
  assert.match(component, /loadResults\(task\.results\s*\|\|\s*\[\],\s*\{\s*collapseSetup:\s*true\s*\}\)/);
  assert.match(component, /:aria-expanded="setupExpanded"/);
  assert.match(component, /setupExpanded\s*=\s*!setupExpanded/);
  assert.match(css, /\.setup-body-shell\.collapsed\s*\{[^}]*grid-template-rows:\s*0fr[^}]*\}/);
  assert.match(css, /\.setup-summary\s*\{[^}]*display:\s*flex[^}]*\}/);
});
