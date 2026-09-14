import test from 'node:test';
import assert from 'node:assert/strict';
import { submitQueuedReview } from '../src/services/reviewQueue.js';
import { mergeReviewTask } from '../src/services/reviewStream.js';

test('HTTP submission happens once; missed events recover through a database read', async () => {
  let submissions = 0;
  let reads = 0;
  let stopped = false;
  const results = await submitQueuedReview({
    payload: { url: 'https://example.com/video.mp4' },
    submit: async () => { submissions++; return { id: 'durable-task' }; },
    read: async () => { if (++reads === 1) throw new Error('temporary disconnect'); return { id: 'durable-task', status: 'completed', results: [{ timestamp: '00:00:01' }] }; },
    stream: () => ({ start() {}, stop() { stopped = true; } }), pollInterval: 5,
  });
  assert.equal(submissions, 1);
  assert.equal(reads, 2);
  assert.equal(stopped, true);
  assert.equal(results.length, 1);
});

test('leaving a queued task stops viewing without cancelling the server task', async () => {
  const controller = new AbortController();
  let stopped = false;
  const pending = submitQueuedReview({
    submit: async () => ({ id: 'queued-task' }),
    read: async () => ({ id: 'queued-task', status: 'queued' }),
    stream: () => ({ start() { controller.abort(); }, stop() { stopped = true; } }),
    signal: controller.signal,
  });
  await assert.rejects(pending, error => error.taskId === 'queued-task');
  assert.equal(stopped, true);
});

test('a new execution attempt resets progress and stale worker events are ignored', () => {
  const old = { id: 'task', status: 'processing', attempt: 1, progress: { attempt: 1, sequence: 30, percent: 80 } };
  const next = { id: 'task', status: 'processing', attempt: 2, progress: { attempt: 2, sequence: 1, percent: 1 } };
  const current = mergeReviewTask(old, next);
  assert.equal(current.progress.percent, 1);
  assert.deepEqual(mergeReviewTask(current, old), current);
});
