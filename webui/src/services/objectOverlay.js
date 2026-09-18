import { normalizeFaceSamples, facesAtTime, faceSampleTimes, faceSampleSeekTime } from './faceOverlay.js';

// Geometry, PTS validation and seeking obey the same contract as face samples.
export const normalizeObjectSamples = (samples, start = 0, end = Infinity) =>
  normalizeFaceSamples(samples, start, end).map(({ similarity, ...sample }) => sample);

// Display padding only: stored/exported coordinates and face overlays stay unchanged.
const expandedObjectBox = ({ x, y, w, h }) => {
  const left = Math.max(0, x - w * .125);
  const top = Math.max(0, y - h * .125);
  return {
    x: left, y: top,
    w: Math.min(1, x + w * 1.125) - left,
    h: Math.min(1, y + h * 1.125) - top
  };
};

export const objectsAtTime = (markers, seconds, category = '') =>
  facesAtTime(markers, seconds, category, 'object_samples')
    .map(object => ({ ...object, originalBox: object.box, box: expandedObjectBox(object.box) }));

// Classify in native video pixels before padding, independently of player zoom.
export function objectDisplayMarker(object, rect, videoSize) {
  const box = object.originalBox ?? object.box;
  if (!['flag', 'logo', 'nudity'].includes(object.objectType)
    || !rect || !videoSize || !(rect.width > 0 && rect.height > 0)
    || !(videoSize.width > 0 && videoSize.height > 0)
    || box.w * videoSize.width > 32 + 1e-9 || box.h * videoSize.height > 32 + 1e-9) return object;
  const markerBox = {
    x: box.x + box.w / 2 - 16 / rect.width,
    y: box.y + box.h / 2 - 16 / rect.height,
    w: 32 / rect.width, h: 32 / rect.height
  };
  // Clip the click target at image edges without shifting the measured center.
  const inset = [
    Math.max(0, -markerBox.y * rect.height),
    Math.max(0, (markerBox.x + markerBox.w - 1) * rect.width),
    Math.max(0, (markerBox.y + markerBox.h - 1) * rect.height),
    Math.max(0, -markerBox.x * rect.width)
  ];
  return { ...object, crosshair: true, box: markerBox, targetClip: `inset(${inset.map(value => `${value}px`).join(' ')})` };
}

export const detectionSampleTimes = (markers, category = '') => [...new Set([
  ...faceSampleTimes(markers, category),
  ...faceSampleTimes(markers, category, 'object_samples')
])].sort((a, b) => a - b);

export const detectionSampleSeekTime = (markers, timeMs, category = '') =>
  faceSampleSeekTime(markers, timeMs, category)
  ?? faceSampleSeekTime(markers, timeMs, category, 'object_samples');
