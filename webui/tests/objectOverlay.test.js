import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResults, serializeResults, filterMarkers } from '../src/services/videoTimeline.js';
import { objectsAtTime, objectDisplayMarker, detectionSampleTimes, detectionSampleSeekTime } from '../src/services/objectOverlay.js';
import { taskProgress } from '../src/services/reviewProgress.js';
import { facesAtTime } from '../src/services/faceOverlay.js';

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

test('object display expands 1.25 times without changing faces or saved evidence', () => {
  const face = { ...row(.4), source: 'face', face_samples: row(.4).object_samples };
  delete face.object_samples;
  const markers = normalizeResults([row(.1), row(.5, 'logo'), row(.8, 'nudity'), face]);
  const original = JSON.stringify(serializeResults(markers));
  const faceBoxes = facesAtTime(markers, 1.0003).map(item => item.box);
  const objects = objectsAtTime(markers, 1.0003);
  for (const [index, x] of [.1, .5, .8].entries()) {
    const box = objects[index].box;
    assert.ok(Math.abs(box.w - .0625) < 1e-12);
    assert.ok(Math.abs(box.h - .1) < 1e-12);
    assert.ok(Math.abs(box.x + box.w / 2 - (x + .025)) < 1e-12);
    assert.ok(Math.abs(box.y + box.h / 2 - .14) < 1e-12);
  }
  assert.deepEqual(faceBoxes, [{ x: .4, y: .1, w: .05, h: .08 }]);
  assert.deepEqual(facesAtTime(markers, 1.0003).map(item => item.box), faceBoxes);
  assert.deepEqual(objectsAtTime(markers, 1.0003), objects); // Re-render never compounds padding.
  assert.equal(JSON.stringify(serializeResults(markers)), original);
});

test('object display padding clips at all image edges', () => {
  const rows = [row(0), row(.75)];
  rows[0].object_samples[0].bbox = { x: 0, y: 0, w: .25, h: .25 };
  rows[1].object_samples[0].bbox = { x: .75, y: .75, w: .25, h: .25 };
  const boxes = objectsAtTime(normalizeResults(rows), 1.0003).map(item => item.box);
  assert.deepEqual(boxes, [
    { x: 0, y: 0, w: .28125, h: .28125 },
    { x: .71875, y: .71875, w: .28125, h: .28125 }
  ]);
});

const videoSize = { width: 1920, height: 1080 };
const playerRect = { left: 50, top: 0, width: 960, height: 540 };
const nativeObject = (width, height, type = 'nudity', x = .3, y = .4) => {
  const finding = row(x, type);
  finding.object_samples[0].bbox = { x, y, w: width / videoSize.width, h: height / videoSize.height };
  return objectsAtTime(normalizeResults([finding]), 1.0003)[0];
};

test('32px threshold uses both native dimensions before the existing display expansion', () => {
  for (const type of ['flag', 'logo', 'nudity']) {
    const small = nativeObject(32, 32, type);
    assert.ok(small.box.w * videoSize.width > 32); // Existing 1.25 padding must not change classification.
    const marker = objectDisplayMarker(small, playerRect, videoSize);
    assert.equal(marker.crosshair, true);
    assert.equal(marker.box.w * playerRect.width, 32);
    assert.equal(marker.box.h * playerRect.height, 32);
    assert.equal(marker.key, small.key);
    assert.equal(marker.candidates, small.candidates);
  }
  for (const [width, height] of [[32.001, 10], [10, 32.001], [100, 12], [42.24, 35.64]]) {
    const object = nativeObject(width, height);
    assert.equal(objectDisplayMarker(object, playerRect, videoSize), object);
  }
});

test('player resizing never changes native size classification or object center', () => {
  const object = nativeObject(24, 28);
  const before = JSON.stringify(object);
  for (const rect of [playerRect, { left: 0, top: 100, width: 1920, height: 1080 }]) {
    const marker = objectDisplayMarker(object, rect, videoSize);
    assert.equal(marker.crosshair, true);
    assert.ok(Math.abs(marker.box.x + marker.box.w / 2 - (object.originalBox.x + object.originalBox.w / 2)) < 1e-12);
    assert.ok(Math.abs(marker.box.y + marker.box.h / 2 - (object.originalBox.y + object.originalBox.h / 2)) < 1e-12);
    assert.equal(marker.box.w * rect.width, 32);
  }
  assert.equal(JSON.stringify(object), before);
});

test('small faces keep their original frames and missing video metadata never guesses a type', () => {
  const object = nativeObject(8, 10);
  const { objectType, originalBox, ...face } = object;
  assert.equal(objectDisplayMarker(face, playerRect, videoSize), face);
  for (const [rect, size] of [[null, videoSize], [playerRect, null], [playerRect, { width: 0, height: 1080 }]]) {
    assert.equal(objectDisplayMarker(object, rect, size), object);
  }
});

test('edge crosshairs stay at the measured center and only their click targets are clipped', () => {
  for (const [x, y] of [[0, 0], [1 - 8 / 1920, 1 - 8 / 1080]]) {
    const object = nativeObject(8, 8, 'logo', x, y);
    const marker = objectDisplayMarker(object, playerRect, videoSize);
    assert.equal(marker.crosshair, true);
    assert.ok(Math.abs(marker.box.x + marker.box.w / 2 - (x + 4 / 1920)) < 1e-12);
    assert.ok(Math.abs(marker.box.y + marker.box.h / 2 - (y + 4 / 1080)) < 1e-12);
    const clip = marker.targetClip.match(/[\d.]+(?=px)/g).map(Number);
    assert.ok(Math.max(...clip) > 0);
    assert.ok(clip.every(value => value >= 0 && value < 16));
  }
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
