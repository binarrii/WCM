import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const component = readFileSync(new URL('../src/views/ParameterConfig.vue', import.meta.url), 'utf8');
const css = readFileSync(new URL('../src/views/parameter-config.css', import.meta.url), 'utf8');
const app = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8');

test('parameter configuration is available from the application menu', () => {
  assert.match(app, /ParameterConfig/);
  assert.match(app, /navigateTo\('parameters'\)/);
  assert.match(app, />参数配置</);
});

test('parameter rows expose expandable formatted JSON details', () => {
  assert.match(component, /expandedKeys\.has\(item\.key\)/);
  assert.match(component, /v-html="highlightJson\(item\.value\)"/);
  assert.match(css, /\.json-viewer/);
  for (const token of ['key', 'string', 'number', 'boolean', 'null']) {
    assert.match(css, new RegExp(`\\.json-${token}`));
  }
});

test('parameter page supports create, update and delete operations', () => {
  assert.match(component, /parameterService\.create/);
  assert.match(component, /parameterService\.update/);
  assert.match(component, /parameterService\.delete/);
});

test('built-in parameters are protected and secrets are never rendered', () => {
  assert.match(component, /:disabled="item\.built_in"/);
  assert.match(component, /form\.builtIn/);
  assert.match(component, /item\.secret/);
  assert.match(component, /敏感值不会通过接口或页面回显/);
  assert.match(component, /item\.has_value \? '••••••••（已配置）'/);
  assert.match(css, /\.parameter-secret/);
  assert.match(css, /\.secret-value/);
});

test('parameter table paginates like review tasks and fills the viewport', () => {
  assert.match(component, /const page = ref\(1\)/);
  assert.match(component, /const pageSize = 30/);
  assert.match(component, /filteredParameters\.value\.slice\(start, start \+ pageSize\)/);
  assert.match(component, /watch\(\[query, selectedGroup\]/);
  assert.match(component, /class="parameter-pagination"/);
  assert.match(component, /第 \{\{ page \}\} \/ \{\{ pageCount \}\} 页/);
  assert.match(css, /\.parameter-config\s*\{[^}]*min-height:\s*calc\(100dvh\s*-\s*108px\)[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.parameter-table-card\s*\{[^}]*flex:\s*1[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.parameter-table-scroll\s*\{[^}]*flex:\s*1[^}]*\}/);
  assert.match(css, /\.parameter-pagination\s*\{[^}]*border-top:/);
});
