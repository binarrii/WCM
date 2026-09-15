import { reactive } from 'vue';
import api, { setCsrfToken } from './api';

export const auth = reactive({ user: null, ready: false, error: '' });
export const roleNames = { superadmin: '超级管理员', admin: '普通管理员', user: '普通用户' };
export const can = permission => Boolean(auth.user?.permissions?.includes(permission));
export function acceptSession(payload) {
  auth.user = payload.user;
  setCsrfToken(payload.csrf_token);
  auth.error = '';
}
export function clearSession() { auth.user = null; setCsrfToken(''); }
export async function refreshSession() {
  try { acceptSession((await api.get('/auth/me')).data); }
  catch (reason) {
    if (reason.response?.status === 401) clearSession();
    else auth.error = '账户服务暂时无法连接，请重试';
  } finally { auth.ready = true; }
}
export async function logout() { await api.post('/auth/logout'); clearSession(); }
export function authError(reason) {
  const detail = reason.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return '请检查输入格式，密码需为 12–128 个字符';
  if (reason.name === 'NotAllowedError') return 'Passkey 操作已取消或超时，请重试';
  if (reason.name === 'InvalidStateError') return '此设备已绑定该 Passkey';
  return reason.message || '操作失败，请重试';
}
