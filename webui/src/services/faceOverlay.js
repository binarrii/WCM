export function normalizeFaceSamples(samples, start = 0, end = Infinity) {
  if (!Array.isArray(samples)) return [];
  return samples.filter(sample => {
    const box = sample?.bbox;
    return Number.isFinite(sample?.time_ms) && sample.time_ms >= start && sample.time_ms <= end
      && box && ['x', 'y', 'w', 'h'].every(key => Number.isFinite(box[key]))
      && box.x >= 0 && box.y >= 0 && box.w > 0 && box.h > 0
      && box.x + box.w <= 1 && box.y + box.h <= 1;
  }).map(sample => {
    const normalized = {
      time_ms: sample.time_ms,
      bbox: { ...sample.bbox },
      similarity: Number.isFinite(sample.similarity) && sample.similarity >= 0 && sample.similarity <= 1
        ? sample.similarity : null
    };
    // Precise PTS must describe the same rounded timestamp, including point findings.
    if (Number.isFinite(sample.pts_seconds) && sample.pts_seconds >= 0
      && Math.abs(sample.pts_seconds * 1000 - sample.time_ms) <= .500001) {
      normalized.pts_seconds = sample.pts_seconds;
      if (Number.isFinite(sample.duration_seconds) && sample.duration_seconds > 0) {
        normalized.duration_seconds = sample.duration_seconds;
      }
      if (Number.isInteger(sample.frame_index) && sample.frame_index >= 0) {
        normalized.frame_index = sample.frame_index;
      }
    }
    return normalized;
  });
}

export function faceSampleSeekTime(markers, timeMs, category = '') {
  if (!Number.isFinite(timeMs) || timeMs < 0) return null;
  const samples = markers.flatMap(marker => marker.findings
    .filter(finding => !category || finding.category === category)
    .flatMap(finding => normalizeFaceSamples(finding.face_samples)))
    .filter(sample => sample.time_ms === timeMs);
  if (!samples.length) return null;
  const sample = samples.find(sample => Number.isFinite(sample.pts_seconds)) || samples[0];
  // Enter the frame's presentation interval. Cap long VFR frames to a small seek offset.
  // Legacy milliseconds can round down by .5 ms; +1 ms crosses that rounding boundary.
  const offset = sample.duration_seconds ? Math.min(sample.duration_seconds / 2, .02) : .001;
  return (sample.pts_seconds ?? sample.time_ms / 1000) + offset;
}

export function faceSampleTimes(markers, category = '') {
  return [...new Set(markers.flatMap(marker => marker.findings
    .filter(finding => !category || finding.category === category)
    .flatMap(finding => normalizeFaceSamples(finding.face_samples).map(sample => sample.time_ms))
  ))].sort((a, b) => a - b);
}

// seconds must be the compositor's mediaTime, never the requested currentTime.
export function facesAtTime(markers, seconds, category = '') {
  const faces = [];
  if (!Number.isFinite(seconds)) return faces;
  // Only tolerate timestamp serialization precision, never a neighbouring video frame.
  const matches = sample => Math.abs((sample.pts_seconds ?? sample.time_ms / 1000) - seconds)
    <= (sample.pts_seconds == null ? .000501 : .00001);
  const candidates = markers.flatMap(marker => marker.findings
    .filter(finding => !category || finding.category === category)
    .flatMap(finding => normalizeFaceSamples(finding.face_samples))).filter(matches);
  const nearest = candidates.sort((a, b) =>
    Math.abs((a.pts_seconds ?? a.time_ms / 1000) - seconds)
      - Math.abs((b.pts_seconds ?? b.time_ms / 1000) - seconds))[0];
  if (!nearest) return faces;
  for (const marker of markers) {
    for (const finding of marker.findings) {
      if (category && finding.category !== category) continue;
      for (const sample of normalizeFaceSamples(finding.face_samples)) {
        const box = sample?.bbox;
        if (sample.time_ms !== nearest.time_ms || !matches(sample)) continue;
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
