import test from 'node:test';
import assert from 'node:assert/strict';
import { observeVideoFrames } from '../src/services/videoFrames.js';
import { facesAtTime } from '../src/services/faceOverlay.js';

class Video extends EventTarget {
  id = 0;
  callbacks = new Map();
  requestVideoFrameCallback(callback) {
    this.callbacks.set(++this.id, callback);
    return this.id;
  }
  cancelVideoFrameCallback(id) { this.callbacks.delete(id); }
  present(mediaTime) {
    const [id, callback] = [...this.callbacks][0];
    this.callbacks.delete(id);
    callback(0, { mediaTime });
  }
}

test('paused cut seek draws only the confirmed frame, even when callback precedes seeked', () => {
  const video = new Video();
  const markers = [{ id: 'cut', findings: [{ description: 'sample', face_samples: [
    { time_ms: 495233, bbox: { x: .6, y: .16, w: .13, h: .31 } }
  ] }] }];
  let faces = [];
  const observer = observeVideoFrames(video, time => { faces = facesAtTime(markers, time); });
  video.present(495.2);
  assert.equal(faces.length, 0);
  const staleCallback = [...video.callbacks.values()][0];
  video.dispatchEvent(new Event('seeking'));
  assert.equal(faces.length, 0);
  staleCallback(0, { mediaTime: 495.233333333 });
  assert.equal(faces.length, 0);
  video.present(495.233333333);
  video.dispatchEvent(new Event('seeked'));
  assert.equal(faces.length, 1);
  video.dispatchEvent(new Event('seeking'));
  video.present(495.266666667);
  assert.equal(faces.length, 0);
  observer.stop();
  assert.equal(video.callbacks.size, 0);
});

test('source replacement and unmount clear frame state and release callbacks', () => {
  const video = new Video();
  let current;
  const observer = observeVideoFrames(video, time => { current = time; });
  video.present(1);
  assert.equal(current, 1);
  video.dispatchEvent(new Event('emptied'));
  assert.equal(current, null);
  const pending = [...video.callbacks.values()][0];
  observer.stop();
  pending(0, { mediaTime: 2 });
  assert.equal(current, null);
  video.dispatchEvent(new Event('seeking'));
  assert.equal(video.callbacks.size, 0);
});

test('unsupported browsers never substitute currentTime for actual frame time', () => {
  const video = new EventTarget();
  video.currentTime = 495.233;
  let current;
  const observer = observeVideoFrames(video, time => { current = time; });
  assert.equal(observer.supported, false);
  assert.equal(current, null);
  video.dispatchEvent(new Event('seeked'));
  assert.equal(current, null);
  observer.stop();
});
