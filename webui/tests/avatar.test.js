import test from 'node:test';
import assert from 'node:assert/strict';
import { avatarInitial, avatarSource, avatarUploadError, MAX_AVATAR_BYTES } from '../src/services/avatar.js';

test('default avatar uses a readable initial and handles missing names', () => {
  assert.equal(avatarInitial({ display_name: ' 张三 ' }), '张');
  assert.equal(avatarInitial({ display_name: ' ', username: 'admin' }), 'A');
  assert.equal(avatarInitial({ display_name: '🦊用户' }), '🦊');
  assert.equal(avatarInitial(null), '?');
});

test('avatar URL uses the configured API and changes with image version', () => {
  const user = { id: 'user/id', avatar_version: 'a'.repeat(64) };
  assert.equal(avatarSource(user, 'https://example.test/api/v1/'), `https://example.test/api/v1/auth/avatars/user%2Fid/${user.avatar_version}`);
  assert.notEqual(avatarSource(user), avatarSource({ ...user, avatar_version: 'b'.repeat(64) }));
  assert.equal(avatarSource({ id: 'user', avatar_version: '../bad' }), '');
  assert.equal(avatarSource({ id: 'user' }), '');
});

test('upload selection rejects unsupported types and oversized images', () => {
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) assert.equal(avatarUploadError({ type, size: MAX_AVATAR_BYTES }), '');
  assert.match(avatarUploadError({ type: 'image/svg+xml', size: 100 }), /JPG/);
  assert.match(avatarUploadError({ type: 'image/png', size: MAX_AVATAR_BYTES + 1 }), /5 MB/);
});
