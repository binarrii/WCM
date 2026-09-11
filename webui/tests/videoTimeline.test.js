import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAnalyzePayload,
  formatTimestamp,
  layoutMarkers,
  markerIsActive,
  normalizeResults,
  filterMarkers,
  serializeResults,
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
  assert.equal(markers.length, 1);
  assert.equal(markers[0].timestamp, '00:00:06.000~00:00:07.000');
  assert.throws(() => normalizeResults({ status: 'error', results: [] }));
});

test('overlapping markers get separate visual lanes', () => {
  const markers = normalizeResults([
    { timestamp: '1~4', category: 'A' },
    { timestamp: '3~5', category: 'B' },
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
    markers.filter(marker => markerIsActive(marker, 6)).flatMap(marker => marker.findings.map(f => f.description)),
    ['甲', '乙']
  );
});

test('controversial Guard verdicts survive normalization and export', () => {
  const [controversial] = normalizeResults([
    { timestamp: '6~7', category: '画面', description: '需人工复核', guard_verdict: 'controversial' }
  ]);

  assert.equal(controversial.findings[0].guard_verdict, 'controversial');
  assert.equal(serializeResults([controversial])[0].guard_verdict, 'controversial');
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


test('same and contained intervals retain distinct evidence in one card and JSON', () => {
  const rows = [
    { timestamp: '55~57', category: '复核', source: 'ocr', description: '字幕一' },
    { timestamp: '55~57', category: '复核', source: 'visual', description: '画面' },
    { timestamp: '56~57', category: '复核', source: 'ocr', description: '字幕二' },
    { timestamp: 56, category: '审核未完成', review_status: 'incomplete', stage: 'face' }
  ];
  const markers = normalizeResults(rows);
  assert.equal(markers.length, 1);
  assert.equal(markers[0].findings.length, 4);
  assert.equal(markers[0].findings[2].timestamp, '00:00:56.000~00:00:57.000');
  const exported = serializeResults(markers);
  assert.equal(exported[0].review_status, 'incomplete');
  assert.deepEqual(normalizeResults(exported), markers);
  const [filtered] = filterMarkers(markers, '审核未完成');
  assert.equal(filtered.timestamp, '00:00:56.000');
  assert.equal(filtered.findings.length, 1);
});

test('nested findings keep precise face PTS and category filtering restores their interval', () => {
  const sample = { time_ms: 2233, pts_seconds: 2 + 7 / 30, duration_seconds: 1 / 30,
    bbox: { x: .2, y: .1, w: .2, h: .3 } };
  const markers = normalizeResults([{ timestamp: '1~5', findings: [
    { timestamp: '1~5', category: '画面', description: '背景' },
    { timestamp: '2~3', category: '人物', description: '甲', face_samples: [sample] }
  ] }]);
  assert.equal(markers.length, 1);
  const filtered = filterMarkers(markers, '人物');
  assert.equal(filtered[0].timestamp, '00:00:02.000~00:00:03.000');
  assert.equal(filtered[0].findings[0].face_samples[0].pts_seconds, sample.pts_seconds);
});

test('legacy one-second gaps are not guessed to belong to one shot', () => {
  assert.equal(normalizeResults([{ timestamp: '15~17' }, { timestamp: '18~19' }]).length, 2);
});
