import test from 'node:test';
import assert from 'node:assert/strict';
import { finishReauthentication, reauthMethods, reauthState, withReauthentication } from '../src/services/reauth.js';

const tick = () => new Promise(resolve => setImmediate(resolve));

test('verification prefers usable Passkey, then bound 2FA, then password', () => {
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: true }, true), ['passkey', 'totp']);
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: false }, true), ['passkey', 'password']);
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: true }, false), ['totp']);
  assert.deepEqual(reauthMethods({ passkeys: [], totp_enabled: false }, false), ['password']);
});

const operation = 'PUT /api/v1/auth/roles/user';

test('every write verifies first, including back-to-back writes', async () => {
  let calls = 0;
  for (const token of ['proof-one', 'proof-two']) {
    const result = withReauthentication(operation, async config => { calls++; return config.headers['X-WCM-Verification']; });
    await tick();
    assert.equal(reauthState.open, true);
    assert.equal(reauthState.operation, operation);
    assert.equal(calls, token === 'proof-one' ? 0 : 1);
    finishReauthentication(token);
    assert.equal(await result, token);
    assert.equal(reauthState.open, false);
  }
  assert.equal(calls, 2);
});

test('cancel never sends the mutation request', async () => {
  let calls = 0;
  const result = withReauthentication(operation, async () => { calls++; });
  const rejected = assert.rejects(result, { code: 'REAUTH_CANCELLED' });
  await tick();
  finishReauthentication();
  await rejected;
  assert.equal(calls, 0);
});

test('concurrent actions cannot share one verification', async () => {
  let secondCalls = 0;
  const first = withReauthentication(operation, async () => 'saved');
  await assert.rejects(withReauthentication('POST /api/v1/auth/password', async () => { secondCalls++; }), { code: 'REAUTH_BUSY' });
  assert.equal(reauthState.operation, operation);
  finishReauthentication('first-proof');
  assert.equal(await first, 'saved');
  assert.equal(secondCalls, 0);
  const second = withReauthentication('POST /api/v1/auth/password', async () => { secondCalls++; });
  await tick();
  assert.equal(reauthState.open, true);
  finishReauthentication('second-proof');
  await second;
  assert.equal(secondCalls, 1);
});

test('write failures never reuse the grant or automatically retry a mutation', async () => {
  for (const status of [400, 403, 500]) {
    let calls = 0;
    const result = withReauthentication(operation, async () => { calls++; throw Object.assign(new Error('failed'), { response: { status } }); });
    const rejected = assert.rejects(result);
    await tick();
    finishReauthentication('single-use-proof');
    await rejected;
    assert.equal(calls, 1);
    assert.equal(reauthState.open, false);
  }
});

test('a boolean success signal cannot release a write without a grant', async () => {
  let called = false;
  const result = withReauthentication(operation, async () => { called = true; });
  const rejected = assert.rejects(result, { code: 'REAUTH_CANCELLED' });
  finishReauthentication(true);
  await rejected;
  assert.equal(called, false);
});
