export function normalizeFaceSamples(samples, start = 0, end = Infinity) {
  if (!Array.isArray(samples)) return [];
  return samples.filter(sample => {
    const box = sample?.bbox;
    return Number.isFinite(sample?.time_ms) && sample.time_ms >= start && sample.time_ms <= end
      && box && ['x', 'y', 'w', 'h'].every(key => Number.isFinite(box[key]))
      && box.x >= 0 && box.y >= 0 && box.w > 0 && box.h > 0
      && box.x + box.w <= 1 && box.y + box.h <= 1;
  }).map(sample => ({
    time_ms: sample.time_ms,
    bbox: { ...sample.bbox },
    similarity: Number.isFinite(sample.similarity) && sample.similarity >= 0 && sample.similarity <= 1
      ? sample.similarity : null
  }));
}

export function faceSampleTimes(markers, category = '') {
  return [...new Set(markers.flatMap(marker => marker.findings
    .filter(finding => !category || finding.category === category)
    .flatMap(finding => normalizeFaceSamples(finding.face_samples).map(sample => sample.time_ms))
  ))].sort((a, b) => a - b);
}

// Only show measured sample positions while paused; never infer a face track.
export function facesAtTime(markers, seconds, category = '') {
  const faces = [];
  // A dense sample sequence must never draw two different frames together.
  const nearestTime = faceSampleTimes(markers, category).reduce((nearest, time) =>
    Math.abs(time - seconds * 1000) < Math.abs(nearest - seconds * 1000) ? time : nearest, Infinity);
  if (Math.abs(nearestTime - seconds * 1000) > 45) return faces;
  for (const marker of markers) {
    for (const finding of marker.findings) {
      if (category && finding.category !== category) continue;
      for (const sample of normalizeFaceSamples(finding.face_samples)) {
        const box = sample?.bbox;
        if (sample.time_ms !== nearestTime) continue;
        const key = [sample.time_ms, box.x, box.y, box.w, box.h].join(':');
        let face = faces.find(item => item.key === key);
        if (!face) {
          face = { key, box, candidates: [] };
          faces.push(face);
        }
        const existing = face.candidates.find(item => item.markerId === marker.id && item.name === finding.description);
        if (!existing) face.candidates.push({ markerId: marker.id, name: finding.description, similarity: sample.similarity });
        else if (sample.similarity != null && (existing.similarity == null || sample.similarity > existing.similarity)) existing.similarity = sample.similarity;
      }
    }
  }
  faces.forEach(face => face.candidates.sort((a, b) => (b.similarity ?? -1) - (a.similarity ?? -1)));
  return faces;
}

export function containedVideoRect(width, height, videoWidth, videoHeight) {
  if (!(width > 0 && height > 0 && videoWidth > 0 && videoHeight > 0)) return null;
  const scale = Math.min(width / videoWidth, height / videoHeight);
  const w = videoWidth * scale;
  const h = videoHeight * scale;
  return { left: (width - w) / 2, top: (height - h) / 2, width: w, height: h };
}
