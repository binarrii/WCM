import assert from 'node:assert/strict';
import test from 'node:test';
import { toSearchRecords } from '../src/services/searchResults.js';

test('one person card with all distinct images and the best score across query faces', () => {
  const hits = [
    { id: 'p1', name: 'Example', face_index: 0, similarity: 0.76,
      image_url: '/cover.jpg', image_urls: ['/cover.jpg', '/2.jpg', '/3.jpg', '/4.jpg'] },
    { id: 'p1', name: 'Example', face_index: 1, similarity: 0.97038, distance: 0.02962,
      face_count: 4, image_url: '/cover.jpg', image_urls: ['/2.jpg'] },
  ];
  const before = JSON.stringify(hits);
  const records = toSearchRecords(hits, 0.3);
  assert.equal(records.length, 1);
  assert.deepEqual(records[0].image_urls, ['/cover.jpg', '/2.jpg', '/3.jpg', '/4.jpg']);
  assert.equal(records[0].image_url, '/cover.jpg');
  assert.equal(records[0].searchSimilarity, 0.97038);
  assert.equal(records[0].searchDistance, 0.02962);
  assert.equal(records[0].face_count, 4);
  assert.equal(JSON.stringify(hits), before);
});

test('filter by exact score, do not merge names, and sort by best similarity', () => {
  const records = toSearchRecords([
    { id: 'p1', name: 'Same name', similarity: 0.3 },
    { id: 'p2', name: 'Same name', similarity: 0.8 },
    { id: 'p3', similarity: 0.29999 },
    { id: 'p4', similarity: 'invalid' },
  ], 0.3);
  assert.deepEqual(records.map(record => record.id), ['p2', 'p1']);
});

test('legacy single images, missing covers and empty results remain supported', () => {
  const records = toSearchRecords([
    { id: 'p1', distance: 0.1, image_url: '/legacy.jpg' },
    { id: 'p2', similarity: 0.7, image_urls: [null, '/available.jpg', '/available.jpg'] },
    { id: 'p3', similarity: 0.5, image_urls: 'invalid' },
  ], 0.3);
  assert.deepEqual(records[0].image_urls, ['/legacy.jpg']);
  assert.equal(records[1].image_url, '/available.jpg');
  assert.equal(records[2].image_url, null);
  assert.deepEqual(records[2].image_urls, []);
  assert.deepEqual(toSearchRecords([], 0.3), []);
});
