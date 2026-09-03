import assert from 'node:assert/strict';
import test from 'node:test';
import { formatImageSimilarity, getImageSimilarity, toSearchRecords } from '../src/services/searchResults.js';

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

test('10% minimum keeps low-similarity results at or above the selected threshold', () => {
  const records = toSearchRecords([
    { id: 'below', similarity: 0.0999 },
    { id: 'boundary', similarity: 0.1 },
    { id: 'above', similarity: 0.2 },
  ], 0.1);
  assert.deepEqual(records.map(record => record.id), ['above', 'boundary']);
});

test('switching gallery images shows their own scores, including below threshold', () => {
  const [record] = toSearchRecords([{
    id: 'p1', similarity: 0.97,
    image_urls: ['/a.jpg', '/b.jpg', '/c.jpg', '/d.jpg'],
    image_similarities: {'/a.jpg': 0.76, '/b.jpg': 0.97, '/c.jpg': 0.22, '/d.jpg': null},
  }], 0.3);
  assert.deepEqual(record.image_urls.map(url => formatImageSimilarity(record, url)),
    ['76%', '97%', '22%', '暂无评分']);
  assert.equal(formatImageSimilarity(record, '/missing.jpg'), '暂无评分');
  assert.equal(getImageSimilarity(record, '/a.jpg'), 0.76);
  assert.equal(record.searchSimilarity, 0.97); // ranking stays unchanged
});

test('scores belong to the best query context, with no fallback to person score', () => {
  const [record] = toSearchRecords([
    {id: 'p1', face_index: 1, similarity: 0.98, image_urls: ['/a.jpg'],
      image_similarities: {'/a.jpg': 0.6}},
    {id: 'p1', face_index: 0, similarity: 0.7, image_urls: ['/a.jpg', '/b.jpg'],
      image_similarities: {'/a.jpg': 0.7, '/b.jpg': 0.5}},
  ], 0.3);
  assert.equal(formatImageSimilarity(record, '/a.jpg'), '60%');
  assert.equal(formatImageSimilarity(record, '/b.jpg'), '暂无评分');
  assert.equal(formatImageSimilarity({image_similarities: {'/a': 0}}, '/a'), '0%');
  assert.equal(formatImageSimilarity({image_similarities: {'/a': NaN}}, '/a'), '暂无评分');
  assert.equal(formatImageSimilarity({searchSimilarity: 0.97}, '/a'), '暂无评分');
});
