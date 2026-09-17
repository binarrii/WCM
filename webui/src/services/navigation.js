const ROUTES = new Set(['home', 'people', 'video', 'tasks', 'media', 'parameters', 'system', 'account', 'users']);

export function routeFromHash(hash) {
  const route = String(hash || '').replace(/^#\/?/, '').split(/[/?]/, 1)[0];
  return ROUTES.has(route) ? route : 'home';
}

export function navigateTo(route) {
  window.location.hash = `#/${ROUTES.has(route) ? route : 'home'}`;
}

export function reviewTaskIdFromHash(hash) {
  const query = String(hash || '').split('?', 2)[1] || '';
  return new URLSearchParams(query).get('task') || '';
}

export function navigateToReviewTask(taskId) {
  window.location.hash = `#/video?task=${encodeURIComponent(taskId)}`;
}
