import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeBase64Url, encodeBase64Url, decodeOptions, serializeCredential } from '../src/services/passkeys.js';

test('WebAuthn binary fields round-trip without losing null bytes or URL alphabet', () => {
  const original = new Uint8Array([0, 1, 127, 128, 254, 255]);
  assert.deepEqual(decodeBase64Url(encodeBase64Url(original)), original);
  const options = decodeOptions({ challenge: 'AP8', user: { id: 'AAE', name: 'member' }, excludeCredentials: [{ id: '_w', type: 'public-key' }] });
  assert.deepEqual(options.challenge, new Uint8Array([0, 255]));
  assert.deepEqual(options.user.id, new Uint8Array([0, 1]));
  assert.deepEqual(options.excludeCredentials[0].id, new Uint8Array([255]));
});

test('credential serialization preserves assertion proof and account handle', () => {
  const raw = new Uint8Array([0, 255]).buffer;
  const result = serializeCredential({ id: 'AP8', rawId: raw, type: 'public-key', response: { clientDataJSON: raw, authenticatorData: raw, signature: raw, userHandle: raw }, getClientExtensionResults: () => ({}) });
  assert.deepEqual(result.response, { clientDataJSON: 'AP8', authenticatorData: 'AP8', signature: 'AP8', userHandle: 'AP8' });
  assert.throws(() => serializeCredential(null), /未取得/);
});
