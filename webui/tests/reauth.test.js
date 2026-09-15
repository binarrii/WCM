import test from 'node:test';
import assert from 'node:assert/strict';
import { finishReauthentication, reauthMethods, reauthState, withReauthentication } from '../src/services/reauth.js';

const required = () => Object.assign(new Error('verification required'), { response: { status: 403, headers: { 'x-wcm-reauth': 'required' } } });
const tick = () => new Promise(resolve => setImmediate(resolve));

test('verification prefers usable Passkey, then bound 2FA, then password', () => {
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: true }, true), ['passkey', 'totp']);
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: false }, true), ['passkey', 'password']);
  assert.deepEqual(reauthMethods({ passkeys: [{}], totp_enabled: true }, false), ['totp']);
  assert.deepEqual(reauthMethods({ passkeys: [], totp_enabled: false }, false), ['password']);
});

test('successful verification resumes the original operation once', async () => {
  let calls = 0;
  const result = withReauthentication(async () => { if (++calls === 1) throw required(); return 'saved'; });
  await tick();
  assert.equal(reauthState.open, true);
  assert.equal(calls, 1);
  finishReauthentication(true);
  assert.equal(await result, 'saved');
  assert.equal(calls, 2);
  assert.equal(reauthState.open, false);
});

test('cancel leaves the protected operation unsubmitted', async () => {
  let calls = 0;
  const result = withReauthentication(async () => { calls++; throw required(); });
  const rejected = assert.rejects(result, { code: 'REAUTH_CANCELLED' });
  await tick();
  finishReauthentication(false);
  await rejected;
  assert.equal(calls, 1);
});

test('concurrent requests share one dialog and are released together', async () => {
  const counts = [0, 0];
  const results = counts.map((_, i) => withReauthentication(async () => { if (++counts[i] === 1) throw required(); return i; }));
  await tick();
  finishReauthentication(true);
  assert.deepEqual(await Promise.all(results), [0, 1]);
  assert.deepEqual(counts, [2, 2]);
});

test('ordinary permission, CSRF and network failures never open the verification dialog', async () => {
  for (const response of [{ status: 403, data: { detail: '权限不足' } }, { status: 403, data: { detail: 'CSRF 无效' } }, { status: 500 }]) {
    let calls = 0;
    await assert.rejects(withReauthentication(async () => { calls++; throw Object.assign(new Error('failed'), { response }); }));
    assert.equal(calls, 1);
    assert.equal(reauthState.open, false);
  }
});

test('a repeated verification rejection does not loop or duplicate further requests', async () => {
  let calls = 0;
  const result = withReauthentication(async () => { calls++; throw required(); });
  const rejected = assert.rejects(result);
  await tick();
  finishReauthentication(true);
  await rejected;
  assert.equal(calls, 2);
  assert.equal(reauthState.open, false);
});
