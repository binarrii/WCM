import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const component = readFileSync(new URL('../src/views/ParameterConfig.vue', import.meta.url), 'utf8');
const css = readFileSync(new URL('../src/views/parameter-config.css', import.meta.url), 'utf8');
const pagination = readFileSync(new URL('../src/components/PaginationBar.vue', import.meta.url), 'utf8');
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

test('parameter editor renders boolean and scalar enum controls', () => {
  assert.match(component, /<option value="boolean">boolean<\/option>/);
  assert.match(component, /<option value="enum">enum<\/option>/);
  assert.match(component, /form\.type === 'boolean'/);
  assert.match(component, /v-model="form\.booleanValue"/);
  assert.match(component, /form\.type === 'enum'/);
  assert.match(component, /parseEnumOptions/);
  assert.match(component, /枚举可选值（仅支持 string、number）/);
  assert.match(component, /v-model="form\.enumValueIndex"/);
  assert.match(css, /\.parameter-type\.boolean/);
  assert.match(css, /\.parameter-type\.enum/);
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
  assert.match(component, /import PaginationBar/);
  assert.match(component, /<PaginationBar :page="page" :page-count="pageCount" :disabled="loading" @change="changePage"/);
  assert.match(css, /\.parameter-config\s*\{[^}]*height:\s*calc\(100dvh\s*-\s*108px\)[^}]*min-height:\s*0[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*overflow:\s*hidden[^}]*\}/);
  assert.match(css, /\.parameter-table-card\s*\{[^}]*flex:\s*1[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*\}/);
  assert.match(css, /\.parameter-table-scroll\s*\{[^}]*min-height:\s*0[^}]*flex:\s*1[^}]*overflow:\s*auto[^}]*\}/);
  assert.doesNotMatch(css, /\.parameter-pagination/);
  assert.match(pagination, /\.pagination-bar\s*\{[^}]*flex:\s*0\s+0\s+auto[^}]*border-top:/);
  assert.match(css, /@media\s*\(max-width:\s*520px\)\s*\{\s*\.parameter-config\s*\{[^}]*height:\s*calc\(100dvh\s*-\s*160px\)/);
});
