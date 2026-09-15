export function passkeyAvailable() {
  return Boolean(window.isSecureContext && window.PublicKeyCredential && navigator.credentials);
}
export function decodeBase64Url(value) {
  const encoded = value.replace(/-/g, '+').replace(/_/g, '/');
  return Uint8Array.from(atob(encoded + '='.repeat((4 - encoded.length % 4) % 4)), c => c.charCodeAt(0));
}
export function encodeBase64Url(value) {
  return btoa(String.fromCharCode(...new Uint8Array(value))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
export function decodeOptions(options) {
  const result = { ...options, challenge: decodeBase64Url(options.challenge) };
  if (options.user) result.user = { ...options.user, id: decodeBase64Url(options.user.id) };
  for (const field of ['allowCredentials', 'excludeCredentials']) {
    if (options[field]) result[field] = options[field].map(item => ({ ...item, id: decodeBase64Url(item.id) }));
  }
  return result;
}
export function serializeCredential(credential) {
  if (!credential) throw new Error('未取得 Passkey，请重试');
  const response = {};
  for (const field of ['clientDataJSON', 'attestationObject', 'authenticatorData', 'signature', 'userHandle']) {
    if (credential.response[field] != null) response[field] = encodeBase64Url(credential.response[field]);
  }
  if (credential.response.getTransports) response.transports = credential.response.getTransports();
  return { id: credential.id, rawId: encodeBase64Url(credential.rawId), type: credential.type,
    response, clientExtensionResults: credential.getClientExtensionResults(), authenticatorAttachment: credential.authenticatorAttachment };
}
export async function createPasskey(options) {
  return serializeCredential(await navigator.credentials.create({ publicKey: decodeOptions(options) }));
}
export async function usePasskey(options, signal) {
  return serializeCredential(await navigator.credentials.get({ publicKey: decodeOptions(options), ...(signal ? { signal } : {}) }));
}
