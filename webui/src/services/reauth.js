import { reactive } from 'vue';

export const reauthState = reactive({ open: false, operation: '' });
let pending = null;
let finish = null;

export function reauthMethods(security, passkeySupported) {
  const methods = [];
  if (passkeySupported && security.passkeys?.length) methods.push('passkey');
  methods.push(security.totp_enabled ? 'totp' : 'password');
  return methods;
}

export function requestReauthentication(operation) {
  if (pending) return Promise.reject(Object.assign(new Error('请先完成当前身份验证'), { code: 'REAUTH_BUSY' }));
  pending = new Promise((resolve, reject) => {
    finish = token => {
      if (typeof token === 'string' && token) resolve(token);
      else reject(Object.assign(new Error('已取消身份验证'), { code: 'REAUTH_CANCELLED' }));
    };
  });
  reauthState.operation = operation;
  reauthState.open = true;
  return pending;
}

export function finishReauthentication(token = null) {
  const complete = finish;
  finish = null;
  pending = null;
  reauthState.open = false;
  reauthState.operation = '';
  complete?.(token);
}

export async function withReauthentication(operation, submit) {
  const token = await requestReauthentication(operation);
  // This grant belongs only to this request; never cache it or automatically retry writes.
  return submit({ headers: { 'X-WCM-Verification': token } });
}
