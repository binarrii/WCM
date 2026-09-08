import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewResultsReady } from '../src/services/reviewStatus.js';
import { normalizeResults } from '../src/services/videoTimeline.js';

test('partial results remain reviewable while unfinished and failed tasks do not', () => {
  for (const status of ['completed', 'partial']) assert.equal(reviewResultsReady(status), true);
  for (const status of ['processing', 'failed', '', undefined]) assert.equal(reviewResultsReady(status), false);
});

test('partial results retain jumpable gaps alongside successful findings at the same timestamp', () => {
  const markers = normalizeResults({ status: 'partial', results: [
    { timestamp: '00:01:14.000', category: '人物', description: '测试人物' },
    { timestamp: '00:01:14.000', category: '审核未完成', description: '视觉审核超时', stage: 'visual', review_status: 'incomplete' },
    { timestamp: '00:01:15.000', category: '审核未完成', description: '文字审核超时', stage: 'ocr', review_status: 'incomplete' }
  ] });
  assert.deepEqual(markers.map(marker => marker.time_ms), [74000, 75000]);
  assert.equal(markers[0].findings.length, 2);
  const gaps = markers.flatMap(marker => marker.findings).filter(finding => finding.review_status === 'incomplete');
  assert.deepEqual(gaps.map(finding => finding.stage), ['visual', 'ocr']);
});
