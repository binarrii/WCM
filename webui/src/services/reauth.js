import { reactive } from 'vue';

export const reauthState = reactive({ open: false });
let pending = null;
let finish = null;

export function reauthMethods(security, passkeySupported) {
  const methods = [];
  if (passkeySupported && security.passkeys?.length) methods.push('passkey');
  methods.push(security.totp_enabled ? 'totp' : 'password');
  return methods;
}

export function requestReauthentication() {
  if (pending) return pending;
  pending = new Promise((resolve, reject) => {
    finish = verified => {
      if (verified) resolve();
      else reject(Object.assign(new Error('已取消身份验证'), { code: 'REAUTH_CANCELLED' }));
    };
  });
  reauthState.open = true;
  return pending;
}

export function finishReauthentication(verified = false) {
  const complete = finish;
  finish = null;
  pending = null;
  reauthState.open = false;
  complete?.(verified);
}

export async function withReauthentication(operation) {
  try { return await operation(); }
  catch (error) {
    const response = error?.response;
    const required = response?.status === 403 && (
      response.headers?.['x-wcm-reauth'] === 'required' ||
      response.data?.detail === '请先重新验证身份，再进行此操作'
    );
    if (!required) throw error;
    await requestReauthentication();
    // Only a specific pre-mutation rejection may be retried, and only once.
    return operation();
  }
}
