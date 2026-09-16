export const MAX_AVATAR_BYTES = 5 * 1024 * 1024;

export function avatarInitial(user) {
  const name = user?.display_name?.trim() || user?.username?.trim() || '?';
  return Array.from(name)[0].toLocaleUpperCase();
}

export function avatarSource(user, apiBase = '/api/v1') {
  if (!user?.id || !/^[a-f0-9]{64}$/.test(user.avatar_version || '')) return '';
  return `${apiBase.replace(/\/$/, '')}/auth/avatars/${encodeURIComponent(user.id)}/${user.avatar_version}`;
}

export function avatarUploadError(file) {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return '请选择 JPG、PNG 或 WebP 图片';
  if (file.size > MAX_AVATAR_BYTES) return '头像图片不能超过 5 MB';
  return '';
}
