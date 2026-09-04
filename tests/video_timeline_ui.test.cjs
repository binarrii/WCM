const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

const template = readFileSync(join(__dirname, '../scripts/video_timeline.html'), 'utf8');
// Execute the template's actual navigation functions, not a duplicate algorithm.
const navigation = template.slice(template.indexOf('function revealEvent('), template.indexOf('function layoutMarkers('));

test('results list stretches through the remaining panel height', () => {
  assert.match(template, /\.results-panel\{[^}]*display:flex[^}]*flex-direction:column[^}]*\}/);
  assert.match(template, /#events\{[^}]*flex:1[^}]*overflow:auto[^}]*\}/);
  assert.match(template, /<aside class="panel results-panel">/);
});

function setup(rows, scrollTop = 0, times = rows.map((_, i) => i * 1000)) {
  const events = {
    clientTop: 2, clientHeight: 200, scrollTop,
    getBoundingClientRect: () => ({top: 100}),
    querySelectorAll: () => [],
  };
  events.children = rows.map(([offset, height]) => ({
    getBoundingClientRect: () => ({top: 102 + offset - events.scrollTop, bottom: 102 + offset + height - events.scrollTop, height}),
    scrollIntoView: () => assert.fail('Must not scroll ancestor containers'),
  }));
  const context = vm.createContext({
    events, visible: times.map(time_ms => ({time_ms})), video: {currentTime: 0},
    updates: 0, update() { context.updates++; },
  });
  vm.runInContext(navigation, context);
  return context;
}

test('jump reveals a row below the list, using only the required scroll', () => {
  const c = setup([[500, 50]], 0, [6000]);
  c.jump(0);
  assert.equal(c.events.scrollTop, 350);
  assert.equal(c.video.currentTime, 6);
  assert.equal(c.updates, 1);
});

test('jump reveals an earlier row and leaves a fully visible row alone', () => {
  const c = setup([[50, 50], [100, 50]], 300);
  c.jump(0);
  assert.equal(c.events.scrollTop, 50);
  c.jump(1);
  assert.equal(c.events.scrollTop, 50);
});

test('fractional row bounds round outward so borders are not clipped', () => {
  const below = setup([[500.08, 50]]);
  below.jump(0);
  assert.equal(below.events.scrollTop, 351);
  const above = setup([[50.9, 50]], 300);
  above.jump(0);
  assert.equal(above.events.scrollTop, 50);
});

test('oversized descriptions reveal their timestamp instead of their bottom', () => {
  const c = setup([[500, 600]]);
  c.jump(0);
  assert.equal(c.events.scrollTop, 500);
  c.events.scrollTop = 650;
  c.jump(0);
  assert.equal(c.events.scrollTop, 500);
});

test('same-start overlapping markers reveal the clicked entry, not the first active row', () => {
  const c = setup([[0, 50], [500, 50]], 0, [6000, 6000]);
  c.jump(1);
  assert.equal(c.events.scrollTop, 350);
  c.jump(0);
  assert.equal(c.events.scrollTop, 0);
});

test('filtered-list indexes, empty lists and invalid navigation remain safe', () => {
  const c = setup([[250, 60]], 0, [99000]);
  c.jump(0);
  assert.equal(c.video.currentTime, 99);
  assert.equal(c.events.scrollTop, 110);
  c.jump(-1); c.jump(1);
  assert.equal(c.updates, 1);
  assert.doesNotThrow(() => setup([]).jump(0));
});

test('playback highlighting does not override manually scrolled results', () => {
  const c = setup([[500, 50]], 30);
  Object.assign(c, {seek: {}, document: {getElementById: () => ({})}, stamp: String,
    timeline: {querySelectorAll: () => []}, prev: {}, next: {}});
  const update = template.match(/function update\(\)\{[\s\S]*?\n\}/)[0];
  vm.runInContext(update, c);
  c.video.currentTime = 0;
  c.update();
  assert.equal(c.events.scrollTop, 30);
});
