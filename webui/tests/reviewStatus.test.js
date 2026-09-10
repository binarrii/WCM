import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewResultsReady, showReviewTaskWarning } from '../src/services/reviewStatus.js';
import { normalizeResults } from '../src/services/videoTimeline.js';

test('partial results remain reviewable while unfinished and failed tasks do not', () => {
  for (const status of ['completed', 'partial']) assert.equal(reviewResultsReady(status), true);
  for (const status of ['processing', 'failed', '', undefined]) assert.equal(reviewResultsReady(status), false);
});

test('a task that fails the coverage threshold still exposes its saved results', () => {
  assert.equal(reviewResultsReady({ status: 'failed', has_results: true }), true);
  assert.equal(reviewResultsReady({ status: 'failed', results: [] }), true);
  assert.equal(reviewResultsReady({ status: 'failed', has_results: false }), false);
  assert.equal(reviewResultsReady({ status: 'processing', has_results: true }), false);
});

test('task-list coverage warnings are hidden below 5%, including just below the boundary', () => {
  for (const [incomplete, total, visible] of [[11, 537, false], [499, 10000, false], [1, 20, true], [3, 20, true]]) {
    const task = {
      status: 'completed', error: '部分采样点未完成',
      review_summary: { total_samples: total, incomplete_samples: incomplete },
      results: [{ timestamp: '00:00:01.000', category: '审核未完成', review_status: 'incomplete' }]
    };
    const original = structuredClone(task);
    assert.equal(showReviewTaskWarning(task), visible);
    assert.deepEqual(task, original);
    assert.equal(reviewResultsReady(task), true);
    assert.equal(normalizeResults(task)[0].findings[0].review_status, 'incomplete');
  }
});

test('missing coverage and execution errors remain visible', () => {
  for (const summary of [undefined, {}, { total_samples: 0, incomplete_samples: 0 }]) {
    assert.equal(showReviewTaskWarning({ status: 'completed', error: '未审核', review_summary: summary }), true);
  }
  assert.equal(showReviewTaskWarning({ status: 'completed' }), false);
  assert.equal(showReviewTaskWarning({
    status: 'failed', error: '任务执行失败',
    review_summary: { total_samples: 537, incomplete_samples: 11 }
  }), true);
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
