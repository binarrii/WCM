import { normalizeFaceSamples, facesAtTime, faceSampleTimes, faceSampleSeekTime } from './faceOverlay.js';

// Geometry, PTS validation and seeking obey the same contract as face samples.
export const normalizeObjectSamples = (samples, start = 0, end = Infinity) =>
  normalizeFaceSamples(samples, start, end).map(({ similarity, ...sample }) => sample);

export const objectsAtTime = (markers, seconds, category = '') =>
  facesAtTime(markers, seconds, category, 'object_samples');

export const detectionSampleTimes = (markers, category = '') => [...new Set([
  ...faceSampleTimes(markers, category),
  ...faceSampleTimes(markers, category, 'object_samples')
])].sort((a, b) => a - b);

export const detectionSampleSeekTime = (markers, timeMs, category = '') =>
  faceSampleSeekTime(markers, timeMs, category)
  ?? faceSampleSeekTime(markers, timeMs, category, 'object_samples');
