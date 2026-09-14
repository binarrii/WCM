import test from 'node:test';
import assert from 'node:assert/strict';
import { paginationItems, parsePageTarget } from '../src/services/pagination.js';

test('short lists expose every page without ellipses', () => {
  assert.deepEqual(paginationItems(1, 1), [1]);
  assert.deepEqual(paginationItems(4, 7), [1, 2, 3, 4, 5, 6, 7]);
});

test('long lists preserve first, current and last pages with bounded controls', () => {
  assert.deepEqual(paginationItems(1, 20), [1, 2, 3, 4, 5, 'end-gap', 20]);
  assert.deepEqual(paginationItems(10, 20), [1, 'start-gap', 9, 10, 11, 'end-gap', 20]);
  assert.deepEqual(paginationItems(20, 20), [1, 'start-gap', 16, 17, 18, 19, 20]);
  for (const count of [8, 9, 20, 100]) {
    for (let page = 1; page <= count; page++) {
      const items = paginationItems(page, count);
      const numbers = items.filter(item => typeof item === 'number');
      assert.equal(items.length, 7);
      assert.equal(new Set(items).size, items.length);
      assert.equal(numbers[0], 1);
      assert.equal(numbers.at(-1), count);
      assert.ok(numbers.includes(page));
      assert.deepEqual(numbers, [...numbers].sort((a, b) => a - b));
    }
  }
  assert.equal(paginationItems(500000, 1000000).length, 7);
});

test('page jumps accept only whole page numbers within the available range', () => {
  assert.equal(parsePageTarget(' 12 ', 20), 12);
  assert.equal(parsePageTarget('01', 20), 1);
  assert.equal(parsePageTarget(20, 20), 20);
  for (const value of ['', ' ', 'abc', '1e1', '1.5', '-1', '0', '21', null, undefined, Infinity]) {
    assert.equal(parsePageTarget(value, 20), null, String(value));
  }
  assert.equal(parsePageTarget('5', 3), null);
});
