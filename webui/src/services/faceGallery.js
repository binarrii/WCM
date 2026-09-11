export const recordImageUrls = record => {
  const values = Array.isArray(record?.image_urls) && record.image_urls.length
    ? record.image_urls
    : record?.image_url ? [record.image_url] : [];
  return [...new Set(values.filter(value => typeof value === 'string' && value))];
};

export const toggleImageDeletion = (images, selected, imageUrl) => {
  const next = new Set(selected);
  if (next.has(imageUrl)) {
    next.delete(imageUrl);
    return { selected: next, blocked: false };
  }
  if (!images.includes(imageUrl) || next.size >= Math.max(0, images.length - 1)) {
    return { selected: next, blocked: true };
  }
  next.add(imageUrl);
  return { selected: next, blocked: false };
};
