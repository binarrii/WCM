import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { recordImageUrls, toggleImageDeletion } from '../src/services/faceGallery.js';

const dashboard = readFileSync(new URL('../src/views/FaceDashboard.vue', import.meta.url), 'utf8');
const dashboardStyles = readFileSync(new URL('../src/views/face-dashboard.css', import.meta.url), 'utf8');

test('recordImageUrls deduplicates galleries and falls back to the cover', () => {
  assert.deepEqual(recordImageUrls({ image_urls: ['/a.jpg', '/a.jpg', '/b.jpg'] }), ['/a.jpg', '/b.jpg']);
  assert.deepEqual(recordImageUrls({ image_url: '/cover.jpg' }), ['/cover.jpg']);
  assert.deepEqual(recordImageUrls({}), []);
});

test('image deletion selection supports one or many but never every image', () => {
  const images = ['/a.jpg', '/b.jpg', '/c.jpg'];
  let result = toggleImageDeletion(images, new Set(), '/a.jpg');
  assert.deepEqual([...result.selected], ['/a.jpg']);
  assert.equal(result.blocked, false);

  result = toggleImageDeletion(images, result.selected, '/b.jpg');
  assert.deepEqual([...result.selected], ['/a.jpg', '/b.jpg']);
  assert.equal(result.blocked, false);

  result = toggleImageDeletion(images, result.selected, '/c.jpg');
  assert.deepEqual([...result.selected], ['/a.jpg', '/b.jpg']);
  assert.equal(result.blocked, true);

  result = toggleImageDeletion(images, result.selected, '/a.jpg');
  assert.deepEqual([...result.selected], ['/b.jpg']);
  assert.equal(result.blocked, false);
});

test('person cards only expose image deletion for multi-image galleries', () => {
  assert.match(dashboard, /v-if="getRecordImages\(record\)\.length > 1"/);
  assert.match(dashboard, /openImageDeleteModal\(record\)/);
  assert.match(dashboard, /v-for="\(imageUrl, index\) in imageDeleteUrls"/);
  assert.match(dashboard, /faceService\.deleteImages\(target\.id, imageUrls\)/);
});

test('image deletion keeps controls visible while the photo grid scrolls', () => {
  assert.match(dashboardStyles, /\.modal-card\.image-delete-modal\s*{[^}]*display:\s*flex;[^}]*overflow:\s*hidden;/s);
  assert.match(dashboardStyles, /\.image-delete-modal \.modal-form\s*{[^}]*min-height:\s*0;[^}]*overflow:\s*hidden;/s);
  assert.match(dashboardStyles, /\.image-delete-grid\s*{[^}]*min-height:\s*0;[^}]*overflow-y:\s*auto;/s);
  assert.match(dashboardStyles, /\.image-delete-modal \.modal-actions\s*{[^}]*flex:\s*0 0 auto;/s);
});
