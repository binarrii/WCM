export const AUTH_BACKGROUND_KEY = 'wcm.auth.background.v1';
export const authBackgrounds = [
  { id: 'optical', name: '深蓝光学', src: new URL('../assets/auth-backgrounds/a-optical-depth.svg', import.meta.url).href },
  { id: 'silver', name: '银白玻璃', src: new URL('../assets/auth-backgrounds/b-silver-frames.svg', import.meta.url).href },
  { id: 'signal', name: '青绿扫描', src: new URL('../assets/auth-backgrounds/c-signal-scan.svg', import.meta.url).href },
  { id: 'flow', name: '靛紫流光', src: new URL('../assets/auth-backgrounds/d-spectral-flow.svg', import.meta.url).href },
];

const isBackground = id => authBackgrounds.some(background => background.id === id);

export function readAuthBackground() {
  try {
    const saved = globalThis.localStorage.getItem(AUTH_BACKGROUND_KEY);
    return isBackground(saved) ? saved : 'random';
  } catch {
    return 'random';
  }
}

export function saveAuthBackground(preference) {
  if (preference !== 'random' && !isBackground(preference)) return false;
  try {
    if (preference === 'random') globalThis.localStorage.removeItem(AUTH_BACKGROUND_KEY);
    else globalThis.localStorage.setItem(AUTH_BACKGROUND_KEY, preference);
    return true;
  } catch {
    return false;
  }
}

export function pickAuthBackground(preference) {
  return authBackgrounds.find(background => background.id === preference)
    || authBackgrounds[Math.floor(Math.random() * authBackgrounds.length)];
}
