// Keep the first/last page and a stable seven-slot window for long lists.
export function paginationItems(page, pageCount) {
  if (pageCount <= 7) return Array.from({ length: pageCount }, (_, index) => index + 1);
  if (page <= 4) return [1, 2, 3, 4, 5, 'end-gap', pageCount];
  if (page >= pageCount - 3) return [1, 'start-gap', pageCount - 4, pageCount - 3, pageCount - 2, pageCount - 1, pageCount];
  return [1, 'start-gap', page - 1, page, page + 1, 'end-gap', pageCount];
}

export function parsePageTarget(value, pageCount) {
  const text = String(value).trim();
  if (!/^\d+$/.test(text)) return null;
  const page = Number(text);
  return Number.isSafeInteger(page) && page >= 1 && page <= pageCount ? page : null;
}
