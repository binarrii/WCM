const ROUTES = new Set(['people', 'video', 'tasks']);

export function routeFromHash(hash) {
  const route = String(hash || '').replace(/^#\/?/, '').split(/[/?]/, 1)[0];
  return ROUTES.has(route) ? route : 'people';
}

export function navigateTo(route) {
  window.location.hash = `#/${ROUTES.has(route) ? route : 'people'}`;
}

export function reviewTaskIdFromHash(hash) {
  const query = String(hash || '').split('?', 2)[1] || '';
  return new URLSearchParams(query).get('task') || '';
}

export function navigateToReviewTask(taskId) {
  window.location.hash = `#/video?task=${encodeURIComponent(taskId)}`;
}
