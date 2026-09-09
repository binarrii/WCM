import test from 'node:test';
import assert from 'node:assert/strict';
import { facesAtTime, containedVideoRect, faceSampleTimes, faceSampleSeekTime } from '../src/services/faceOverlay.js';
import { normalizeResults } from '../src/services/videoTimeline.js';

test('groups candidates and only renders measured sample times', () => {
  const bbox = { x: .2, y: .1, w: .15, h: .3 };
  const markers = normalizeResults(['甲', '乙'].map(description => ({
    timestamp: '00:00:01.000~00:00:02.000', category: '人物', description,
    face_samples: [{ time_ms: 1000, bbox, similarity: .85 }]
  })));
  assert.equal(facesAtTime(markers, 1).length, 1);
  assert.equal(facesAtTime(markers, 1)[0].candidates.length, 2);
  assert.equal(facesAtTime(markers, 1.5).length, 0);
  assert.equal(facesAtTime(markers, 1, '其他').length, 0);
});
test('legacy and malformed positions do not render', () => {
  const markers = normalizeResults([{ timestamp: 1, description: '甲', face_samples: [
    { time_ms: 1000, bbox: { x: .9, y: 0, w: .5, h: .2 } }, null
  ] }]);
  assert.deepEqual(facesAtTime(markers, 1), []);
  assert.deepEqual(facesAtTime(normalizeResults([{ timestamp: 1 }]), 1), []);
});
test('accounts for pillarbox and letterbox offsets', () => {
  assert.deepEqual(containedVideoRect(1000, 600, 800, 600), { left: 100, top: 0, width: 800, height: 600 });
  assert.deepEqual(containedVideoRect(800, 600, 1600, 800), { left: 0, top: 100, width: 800, height: 400 });
  assert.equal(containedVideoRect(800, 600, 0, 0), null);
});

test('dense samples use only the closest measured frame', () => {
  const markers = normalizeResults([{ timestamp: '00:00:01.000~00:00:02.000', face_samples: [
    { time_ms: 1000, bbox: { x: .1, y: .1, w: .1, h: .1 } },
    { time_ms: 1040, bbox: { x: .6, y: .1, w: .1, h: .1 } }
  ] }]);
  assert.equal(facesAtTime(markers, 1.03).length, 0);
  assert.equal(facesAtTime(markers, 1.04)[0].box.x, .6);
  assert.deepEqual(faceSampleTimes(markers), [1000, 1040]);
});
test('invalid imported samples cannot redirect a seek outside the finding interval', () => {
  const bbox = { x: .1, y: .1, w: .2, h: .2 };
  const markers = normalizeResults([{ timestamp: '00:00:01.000~00:00:02.000', face_samples: [
    { time_ms: -1, bbox }, { time_ms: 3000, bbox }, { time_ms: 1500, bbox, similarity: 999 }
  ] }]);
  assert.deepEqual(faceSampleTimes(markers), [1500]);
  assert.equal(facesAtTime(markers, 1.5)[0].candidates[0].similarity, null);
});
test('duplicate candidates retain their best similarity', () => {
  const bbox = { x: .1, y: .1, w: .2, h: .2 };
  const markers = normalizeResults([{ timestamp: 1, description: '甲', face_samples: [
    { time_ms: 1000, bbox, similarity: .6 }, { time_ms: 1000, bbox, similarity: .9 }
  ] }]);
  assert.equal(facesAtTime(markers, 1)[0].candidates[0].similarity, .9);
});

const cutSample = extra => normalizeResults([{
  timestamp: '00:08:15.233', description: '测试人物',
  face_samples: [{ time_ms: 495233, bbox: { x: .598, y: .162, w: .128, h: .310 }, ...extra }]
}]);

test('rounded legacy cut timestamp seeks after the boundary and never shows on the previous frame', () => {
  const markers = cutSample({});
  assert.equal(faceSampleSeekTime(markers, 495233), 495.234);
  assert.equal(facesAtTime(markers, 495.2).length, 0);
  assert.equal(facesAtTime(markers, 495.233333333).length, 1);
  assert.equal(facesAtTime(markers, 495.266666667).length, 0);
  assert.equal(facesAtTime(markers, null).length, 0);
});

test('precise PTS survives a rounded point interval and seeks inside the actual frame', () => {
  const pts = 495 + 7 / 30;
  const markers = cutSample({ pts_seconds: pts, duration_seconds: 1 / 30, frame_index: 14857 });
  const sample = markers[0].findings[0].face_samples[0];
  assert.equal(sample.pts_seconds, pts);
  assert.equal(sample.frame_index, 14857);
  assert.ok(Math.abs(faceSampleSeekTime(markers, 495233) - 495.25) < 1e-9);
  assert.equal(facesAtTime(markers, pts).length, 1);
  assert.equal(facesAtTime(markers, 495.233).length, 0); // Requested time is not the presented PTS.
});

test('VFR frame seek offset respects even a short frame duration', () => {
  const pts = 495.2334;
  const markers = cutSample({ pts_seconds: pts, duration_seconds: .0004 });
  const seek = faceSampleSeekTime(markers, 495233);
  assert.ok(seek > pts && seek < pts + .0004);
  assert.equal(facesAtTime(markers, pts + .0004).length, 0);
});

test('untrusted precise metadata cannot redirect a seek to another timestamp', () => {
  const markers = cutSample({ pts_seconds: 900, duration_seconds: 10000, frame_index: -1 });
  assert.equal(faceSampleSeekTime(markers, 495233), 495.234);
  assert.equal(faceSampleSeekTime(markers, -1), null);
  assert.equal(faceSampleSeekTime(markers, 999999), null);
  assert.equal(markers[0].findings[0].face_samples[0].pts_seconds, undefined);
});
