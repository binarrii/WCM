import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResults, serializeResults, filterMarkers } from '../src/services/videoTimeline.js';
import { objectsAtTime, detectionSampleTimes, detectionSampleSeekTime } from '../src/services/objectOverlay.js';
import { taskProgress } from '../src/services/reviewProgress.js';

const row = (x, type = 'flag') => ({
  timestamp: '00:00:01.000', source: 'flags', category: '旗帜与徽标',
  name: '测试目标', description: '测试目标', object_type: type, review_status: 'needs_review',
  object_samples: [{ time_ms: 1000, pts_seconds: 1.0003, duration_seconds: .04, frame_index: 25,
    bbox: { x, y: .1, w: .05, h: .08 } }]
});

test('all instances and precise metadata survive grouping, filtering and JSON round trip', () => {
  const markers = normalizeResults([row(.1), row(.5), row(.8, 'logo')]);
  const roundTrip = normalizeResults(JSON.parse(JSON.stringify(serializeResults(markers))));
  const filtered = filterMarkers(roundTrip, '旗帜与徽标');
  const objects = objectsAtTime(filtered, 1.0003);
  assert.equal(objects.length, 3);
  assert.deepEqual(objects.map(item => item.objectType), ['flag', 'flag', 'logo']);
  assert.equal(objects[0].candidates[0].needsReview, true);
  assert.equal(objects[0].candidates[0].similarity, null);
  assert.deepEqual(detectionSampleTimes(filtered), [1000]);
  assert.equal(detectionSampleSeekTime(filtered, 1000), 1.0203);
  assert.equal(filtered[0].findings[0].object_samples[0].frame_index, 25);
  assert.ok(filtered[0].findings.every(f => f.review_status === 'needs_review'));
  assert.deepEqual(objectsAtTime(filtered, 1.04), []);
  assert.deepEqual(objectsAtTime(filtered, null), []);
  assert.deepEqual(objectsAtTime(filtered, 1.0003, '其他类别'), []);
});

test('invalid imported boxes or times cannot draw or redirect the player', () => {
  const invalid = row(.99);
  invalid.object_samples.push({ time_ms: 3000, pts_seconds: 3, bbox: { x: .1, y: .1, w: .1, h: .1 } });
  const markers = normalizeResults([invalid]);
  assert.deepEqual(detectionSampleTimes(markers), []);
  assert.equal(detectionSampleSeekTime(markers, 3000), null);
});

test('pending flags are a successful review stage and progress uses a readable label', () => {
  const progress = taskProgress({ status: 'processing', progress: { phase: 'reviewing', active_windows: [
    { index: 1, start_seconds: 0, end_seconds: 1, sample_timestamps: [1], stages: { flags: [1] } }
  ] } });
  assert.match(progress.windows[0].stages, /对象检测.*裸露/);
});

test('nudity has its own filter and preserves scope and evidence through JSON and frame seeking', () => {
  const nude = { ...row(.3, 'nudity'), category: '裸露部位', name: '生殖器官裸露',
    object_target: 'exposed_genitals', object_evidence: '实际可见的未遮盖部位' };
  const markers = normalizeResults([row(.1), nude]);
  const restored = normalizeResults(JSON.parse(JSON.stringify(serializeResults(markers))));
  const filtered = filterMarkers(restored, '裸露部位');
  const objects = objectsAtTime(filtered, 1.0003);
  assert.equal(objects.length, 1);
  assert.equal(objects[0].objectType, 'nudity');
  assert.equal(objects[0].candidates[0].needsReview, true);
  assert.equal(filtered[0].findings[0].object_target, nude.object_target);
  assert.equal(filtered[0].findings[0].object_evidence, nude.object_evidence);
  assert.deepEqual(detectionSampleTimes(filtered), [1000]);
  assert.deepEqual(objectsAtTime(filtered, 1.04), []);
});
