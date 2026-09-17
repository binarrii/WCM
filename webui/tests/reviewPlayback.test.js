import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewPlayback, playbackErrorMessage } from '../src/services/reviewPlayback.js';

test('ready media uses the immutable version and video clock offset', () => {
  const media = { url: '/api/v1/review_tasks/t/media/abc', expires_at: '2099-01-01T00:00:00Z', video_start_seconds: 0.144 };
  assert.deepEqual(reviewPlayback({ media }, 'http://source/v.ts'), { url: media.url, offset: .144, message: '' });
});
test('expired media never silently falls back to a changed original', () => {
  const state = reviewPlayback({ media: { expires_at: '2000-01-01T00:00:00Z' } }, 'http://source/v.mp4');
  assert.equal(state.url, '');
  assert.match(state.message, /保存期限/);
});
test('preparation and unsupported historical sources explain their state', () => {
  assert.match(reviewPlayback({ status: 'processing' }, 'http://source/v.mp4').message, /准备/);
  assert.match(reviewPlayback(null, 'http://source/v.ts?token=1').message, /新的审核任务/);
  assert.equal(reviewPlayback(null, 'http://source/v.mp4?token=1').url, 'http://source/v.mp4?token=1');
  assert.match(playbackErrorMessage({ code: 3 }), /解码/);
  assert.doesNotMatch(playbackErrorMessage({ code: 4 }), /Range/);
});
