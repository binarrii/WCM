import api from './api';
import { buildAnalyzePayload } from './videoTimeline';
import { submitQueuedReview } from './reviewQueue';
import { reviewTaskService } from './reviewTaskService';

export const mediaService = {
  async analyzeVideo({ url, sampleInterval = 1, topK = 10, minSimilarity = 0.5, onTaskAccepted, onTaskEvent, signal }) {
    const payload = buildAnalyzePayload({
      url, sampleInterval, topK, minSimilarity
    });
    if (onTaskAccepted) return submitQueuedReview({
      payload, onTaskAccepted, onTaskEvent, signal,
      submit: async (body, signal) => (await api.post('/review_tasks', body, { signal })).data,
      read: id => reviewTaskService.get(id),
      stream: callbacks => reviewTaskService.stream(callbacks),
    });
    const response = await api.post('/analyze_media', payload);
    return response.data;
  }
};
