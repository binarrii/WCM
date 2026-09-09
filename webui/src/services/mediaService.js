import api, { API_BASE } from './api';
import { buildAnalyzePayload } from './videoTimeline';
import { reviewSocketUrl, submitReview } from './reviewSubmission';

export const mediaService = {
  async analyzeVideo({ url, sampleInterval = 1, topK = 10, minSimilarity = 0.5, onTaskAccepted, onTaskEvent, signal }) {
    const payload = buildAnalyzePayload({
      url, sampleInterval, topK, minSimilarity
    });
    if (onTaskAccepted) return submitReview(reviewSocketUrl(API_BASE, window.location.href), payload, onTaskAccepted, undefined, onTaskEvent, signal);
    const response = await api.post('/analyze_media', payload);
    return response.data;
  }
};
