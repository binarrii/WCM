import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';
export const IMAGE_BASE = import.meta.env.VITE_IMAGE_BASE || '';

const api = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
  headers: { 'X-WCM-Client': 'web' }
});

let csrfToken = '';
export function setCsrfToken(value) { csrfToken = value || ''; }
api.interceptors.request.use(config => {
  if (csrfToken) config.headers['X-CSRF-Token'] = csrfToken;
  return config;
});
api.interceptors.response.use(response => response, error => {
  if (error.response?.status === 401 && !/\/auth\/(login|register|passkeys\/login)/.test(error.config?.url || '')) {
    window.dispatchEvent(new Event('wcm-session-expired'));
  }
  return Promise.reject(error);
});

export default api;
