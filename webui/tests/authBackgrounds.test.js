import test from 'node:test';
import assert from 'node:assert/strict';
import { AUTH_BACKGROUND_KEY, authBackgrounds, pickAuthBackground, readAuthBackground, saveAuthBackground } from '../src/services/authBackgrounds.js';

function storageFor(t, initial = {}) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const values = new Map(Object.entries(initial));
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true });
  t.after(() => { if (original) Object.defineProperty(globalThis, 'localStorage', original); else delete globalThis.localStorage; });
  return { storage, values };
}

test('opening without a saved choice can select all four backgrounds without pinning one', t => {
  const { values } = storageFor(t);
  const random = t.mock.method(Math, 'random');
  const seen = [];
  for (const value of [0, .25, .5, .999999]) {
    random.mock.mockImplementation(() => value);
    assert.equal(readAuthBackground(), 'random');
    seen.push(pickAuthBackground(readAuthBackground()).id);
  }
  assert.deepEqual(seen, authBackgrounds.map(background => background.id));
  assert.equal(values.has(AUTH_BACKGROUND_KEY), false);
});

test('a manual choice stays fixed on subsequent opens regardless of randomness', t => {
  storageFor(t);
  t.mock.method(Math, 'random', () => { throw new Error('Pinned background must not be random'); });
  for (const background of authBackgrounds) {
    assert.equal(saveAuthBackground(background.id), true);
    assert.equal(readAuthBackground(), background.id);
    assert.equal(pickAuthBackground(readAuthBackground()).id, background.id);
  }
});

test('restoring random removes only the background preference', t => {
  const { values } = storageFor(t, { [AUTH_BACKGROUND_KEY]: 'silver', theme: 'dark' });
  assert.equal(saveAuthBackground('random'), true);
  assert.equal(readAuthBackground(), 'random');
  assert.equal(values.has(AUTH_BACKGROUND_KEY), false);
  assert.equal(values.get('theme'), 'dark');
});

test('an obsolete or invalid saved value falls back to a supported random background', t => {
  const { values } = storageFor(t, { [AUTH_BACKGROUND_KEY]: 'obsolete' });
  assert.equal(readAuthBackground(), 'random');
  assert.ok(authBackgrounds.includes(pickAuthBackground(readAuthBackground())));
  assert.equal(saveAuthBackground('invalid'), false);
  assert.equal(values.get(AUTH_BACKGROUND_KEY), 'obsolete');
});

test('blocked browser storage does not prevent rendering or switching backgrounds', t => {
  storageFor(t);
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get() { throw new Error('Storage blocked'); } });
  assert.equal(readAuthBackground(), 'random');
  assert.equal(saveAuthBackground('optical'), false);
  assert.equal(saveAuthBackground('random'), false);
  assert.equal(pickAuthBackground('silver').id, 'silver');
});
