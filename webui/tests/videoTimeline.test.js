import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAnalyzePayload,
  formatTimestamp,
  layoutMarkers,
  markerIsActive,
  normalizeResults,
  similarityToDistance,
  timestampMs,
  validateVideoUrl
} from '../src/services/videoTimeline.js';

test('timestamps round-trip points and ranges', () => {
  assert.equal(timestampMs('01:02:03.045'), 3723045);
  assert.equal(formatTimestamp(3723045), '01:02:03.045');
  assert.throws(() => timestampMs('00:60:00'));
  const markers = normalizeResults([
    { timestamp: '00:00:06.000~00:00:07.000', category: '人物', description: '甲' },
    { timestamp: '00:00:06.000~00:00:07.000', category: '人物', description: '甲' },
    { timestamp: 6, category: '文本', description: '待复核' }
  ]);
  assert.equal(markers.length, 2);
  assert.equal(markers[1].timestamp, '00:00:06.000~00:00:07.000');
  assert.throws(() => normalizeResults({ status: 'error', results: [] }));
});

test('overlapping markers get separate visual lanes', () => {
  const markers = normalizeResults([
    { timestamp: '1~4', category: 'A' },
    { timestamp: '2~3', category: 'B' },
    { timestamp: 8, category: 'C' }
  ]);
  const layout = layoutMarkers(markers, 10000, 1000);
  assert.deepEqual(layout.map(item => item.lane), [0, 1, 0]);
});

test('active state covers ranges and point tolerance', () => {
  const [range, point] = normalizeResults([
    { timestamp: '1~2', category: 'A' },
    { timestamp: 3, category: 'B' }
  ]);
  assert.equal(markerIsActive(range, 1.5), true);
  assert.equal(markerIsActive(point, 3.05), true);
  assert.equal(markerIsActive(point, 3.2), false);
});

test('all findings at the clicked time become active together', () => {
  const markers = normalizeResults([
    { timestamp: '6~8', category: '人物', description: '甲' },
    { timestamp: '6~7', category: '人物', description: '乙' },
    { timestamp: '9~10', category: '人物', description: '丙' }
  ]);
  assert.deepEqual(
    markers.filter(marker => markerIsActive(marker, 6)).map(marker => marker.findings[0].description),
    ['乙', '甲']
  );
});

test('minimum similarity maps to the legacy distance contract', () => {
  assert.equal(similarityToDistance(0.5), 0.5);
  assert.equal(similarityToDistance(0.1), 0.9);
  assert.equal(similarityToDistance(1), 0.000001);
  assert.throws(() => similarityToDistance(0.09));
});

test('analysis request uses API field names and converts similarity to distance', () => {
  assert.deepEqual(buildAnalyzePayload({
    url: 'http://example.com/video.mp4', sampleInterval: 2, topK: 5, minSimilarity: 0.7
  }), {
    url: 'http://example.com/video.mp4', sample_interval: 2, top_k: 5, threshold: 0.3
  });
  assert.throws(() => buildAnalyzePayload({ url: 'http://example.com/a.mp4', sampleInterval: 0 }));
  assert.throws(() => buildAnalyzePayload({ url: 'http://example.com/a.mp4', topK: 11 }));
});

test('only HTTP video addresses are accepted', () => {
  assert.equal(validateVideoUrl('http://example.com/video.mp4'), 'http://example.com/video.mp4');
  assert.throws(() => validateVideoUrl('file:///tmp/video.mp4'));
  assert.throws(() => validateVideoUrl('not a URL'));
});
