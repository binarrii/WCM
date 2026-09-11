import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatParameterValue,
  highlightJson,
  parseEnumOptions,
  parseParameterValue,
  summarizeParameterValue
} from '../src/services/parameterFormatting.js';

test('JSON values are formatted and syntax highlighted by token type', () => {
  const value = { enabled: true, retries: 3, label: '审核', fallback: null };
  assert.equal(formatParameterValue(value, 'json'), JSON.stringify(value, null, 2));
  const highlighted = highlightJson(value);
  assert.match(highlighted, /class="json-key">"enabled"/);
  assert.match(highlighted, /class="json-boolean">true/);
  assert.match(highlighted, /class="json-number">3/);
  assert.match(highlighted, /class="json-string">"审核"/);
  assert.match(highlighted, /class="json-null">null/);
});

test('JSON highlighting escapes markup before adding trusted token spans', () => {
  const highlighted = highlightJson({ html: '<img src=x onerror=alert(1)>' });
  assert.doesNotMatch(highlighted, /<img/);
  assert.match(highlighted, /&lt;img src=x onerror=alert\(1\)&gt;/);
});

test('editor parsing enforces selected types', () => {
  assert.equal(parseParameterValue('42.5', 'number'), 42.5);
  assert.deepEqual(parseParameterValue('{"a":[1]}', 'json'), { a: [1] });
  assert.equal(parseParameterValue('{raw}', 'string'), '{raw}');
  assert.equal(parseParameterValue('false', 'boolean'), false);
  assert.equal(formatParameterValue(true, 'boolean'), 'true');
  assert.equal(formatParameterValue('auto', 'enum'), 'auto');
  assert.throws(() => parseParameterValue('"42"', 'number'), /有效数字/);
  assert.throws(() => parseParameterValue('0', 'boolean'), /true 或 false/);
  assert.throws(() => parseParameterValue('{bad}', 'json'), /有效 JSON/);
});

test('enum options only accept unique string and finite number values', () => {
  assert.deepEqual(parseEnumOptions('["auto", 1, 2.5]'), ['auto', 1, 2.5]);
  assert.throws(() => parseEnumOptions('[]'), /非空/);
  assert.throws(() => parseEnumOptions('[true]'), /string 或 number/);
  assert.throws(() => parseEnumOptions('[null]'), /string 或 number/);
  assert.throws(() => parseEnumOptions('[{}]'), /string 或 number/);
  assert.throws(() => parseEnumOptions('["auto", "auto"]'), /不能重复/);
  assert.throws(() => parseEnumOptions('[1, 1.0]'), /不能重复/);
});

test('collapsed values are single-line and bounded', () => {
  assert.equal(summarizeParameterValue({ a: [1, 2] }, 'json'), '{"a":[1,2]}');
  assert.equal(summarizeParameterValue('abcdef', 'string', 4), 'abcd…');
});
