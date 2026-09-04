import test from 'node:test';
import assert from 'node:assert/strict';
import { routeFromHash } from '../src/services/navigation.js';

test('sidebar routes are stable across refreshes', () => {
  assert.equal(routeFromHash('#/people'), 'people');
  assert.equal(routeFromHash('#/video'), 'video');
  assert.equal(routeFromHash('#/video?source=remote'), 'video');
});

test('unknown and empty routes fall back to people management', () => {
  assert.equal(routeFromHash(''), 'people');
  assert.equal(routeFromHash('#/unknown'), 'people');
});
