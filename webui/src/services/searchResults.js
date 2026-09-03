export const getSearchSimilarity = (item) => Number(
  item.similarity ?? (1 - (item.distance ?? 1))
);

export const getImageSimilarity = (record, imageUrl) => {
  const value = record.image_similarities?.[imageUrl];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

export const formatImageSimilarity = (record, imageUrl) => {
  const value = getImageSimilarity(record, imageUrl);
  return value === null ? '暂无评分' : `${(value * 100).toFixed(0)}%`;
};

// The API retains separate query-face contexts. The dashboard displays people,
// so combine those contexts into one card, scored by the person's best hit.
export const toSearchRecords = (results, minSimilarity) => {
  const people = new Map();
  results.forEach((item, index) => {
    const similarity = getSearchSimilarity(item);
    if (!Number.isFinite(similarity) || similarity < minSimilarity) return;
    const key = item.id || `unknown-${index}`;
    const previous = people.get(key);
    const urls = [...new Set([
      ...(previous?.image_urls || []),
      item.image_url,
      ...(Array.isArray(item.image_urls) ? item.image_urls : []),
    ].filter(url => typeof url === 'string' && url.length > 0))];
    if (!previous || similarity > previous.searchSimilarity) {
      people.set(key, {
        id: item.id,
        name: item.name,
        created_at: item.created_at,
        face_count: item.face_count,
        // Keep scores from the selected query face, never combine scores
        // from different faces or apply the person's maximum to all images.
        image_similarities: item.image_similarities || {},
        person: {
          name: item.name,
          occupation: item.occupation,
          type: item.type,
          remarks: item.remarks,
        },
        searchDistance: item.distance,
        searchSimilarity: similarity,
      });
    }
    people.get(key).image_urls = urls;
    people.get(key).image_url = urls[0] || null;
  });
  return [...people.values()].sort((a, b) => b.searchSimilarity - a.searchSimilarity);
};
