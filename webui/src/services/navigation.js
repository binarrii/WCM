const ROUTES = new Set(['people', 'video']);

export function routeFromHash(hash) {
  const route = String(hash || '').replace(/^#\/?/, '').split(/[/?]/, 1)[0];
  return ROUTES.has(route) ? route : 'people';
}

export function navigateTo(route) {
  window.location.hash = `#/${ROUTES.has(route) ? route : 'people'}`;
}
