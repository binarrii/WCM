import api from './api';
import { buildAnalyzePayload } from './videoTimeline';

export const mediaService = {
  async analyzeVideo({ url, sampleInterval = 1, topK = 10, minSimilarity = 0.5 }) {
    const response = await api.post('/analyze_media', buildAnalyzePayload({
      url, sampleInterval, topK, minSimilarity
    }));
    return response.data;
  }
};
