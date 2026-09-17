import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewTaskIdFromHash, routeFromHash } from '../src/services/navigation.js';

test('sidebar routes are stable across refreshes', () => {
  assert.equal(routeFromHash('#/home'), 'home');
  assert.equal(routeFromHash('#/people'), 'people');
  assert.equal(routeFromHash('#/video'), 'video');
  assert.equal(routeFromHash('#/video?source=remote'), 'video');
  assert.equal(routeFromHash('#/tasks'), 'tasks');
  assert.equal(routeFromHash('#/media'), 'media');
  assert.equal(routeFromHash('#/parameters'), 'parameters');
  assert.equal(routeFromHash('#/system'), 'system');
});

test('review task id is read from the video route query', () => {
  assert.equal(reviewTaskIdFromHash('#/video?task=task-123'), 'task-123');
  assert.equal(reviewTaskIdFromHash('#/video?task=%E6%B5%8B%E8%AF%95'), '测试');
  assert.equal(reviewTaskIdFromHash('#/video'), '');
});

test('unknown and empty routes open the home workspace', () => {
  assert.equal(routeFromHash(''), 'home');
  assert.equal(routeFromHash('#/unknown'), 'home');
});
