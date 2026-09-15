import test from 'node:test';
import assert from 'node:assert/strict';
import { canTriggerSync, manualRequestLabel, replicaPresentation, formatSyncTime } from '../src/services/replicationStatus.js';

const enabled = { enabled: true, initialized: true, primary_recovering: false };
test('only ready and retry replicas allow manual sync; recovery is never a button action', () => {
  for (const state of ['new', 'disabled', 'quarantined', 'draining', 'syncing', 'unknown']) {
    assert.equal(canTriggerSync(enabled, { state }), false);
  }
  for (const state of ['ready', 'retry']) {
    assert.equal(canTriggerSync(enabled, { state }), true);
    assert.equal(canTriggerSync({ ...enabled, primary_recovering: true }, { state }), false);
    assert.equal(canTriggerSync({ ...enabled, initialized: false }, { state }), false);
  }
});
test('ready state alone does not imply that a replica can serve reads', () => {
  assert.equal(replicaPresentation({ state: 'ready', heartbeat_fresh: false }).label, '心跳过期');
  assert.equal(replicaPresentation({ state: 'ready', heartbeat_fresh: true, lag: 3 }).label, '等待追平');
  assert.equal(replicaPresentation({ state: 'ready', read_eligible: true }).tone, 'success');
});
test('accepted requests remain pending until worker acknowledgement', () => {
  const request = { id: 'r', completed_at: null };
  assert.equal(manualRequestLabel({ state: 'ready', manual_request: request }), '等待同步进程处理');
  assert.equal(manualRequestLabel({ state: 'quarantined', manual_request: request }), '等待人工处理');
  assert.equal(manualRequestLabel({ state: 'retry', manual_request: request }), '未完成 · 等待重试');
  assert.equal(manualRequestLabel({ state: 'ready', manual_request: { ...request, completed_at: '2026-09-15T03:00:00Z' } }), '已完成');
});
test('UTC timestamps remain consistent during rolling API upgrades', () => {
  assert.equal(formatSyncTime('2026-09-15T03:00:00'), formatSyncTime('2026-09-15T03:00:00Z'));
  assert.equal(formatSyncTime('2026-09-15T03:00:00+00:00'), formatSyncTime('2026-09-15T03:00:00Z'));
  assert.equal(formatSyncTime(null), '—');
});
