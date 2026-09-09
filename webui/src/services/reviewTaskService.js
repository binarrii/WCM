import api, { API_BASE } from './api';
import { createReviewStream, reviewStreamUrl } from './reviewStream';

export const reviewTaskService = {
  stream(callbacks) {
    return createReviewStream({ url: reviewStreamUrl(API_BASE, window.location.href), ...callbacks });
  },
  async list({ query = '', status = '', page = 1, pageSize = 30 } = {}) {
    const response = await api.get('/review_tasks', {
      params: { q: query || undefined, status: status || undefined, page, page_size: pageSize }
    });
    return response.data;
  },

  async get(taskId) {
    const response = await api.get(`/review_tasks/${encodeURIComponent(taskId)}`);
    return response.data;
  },

  async downloadResults(taskId) {
    const response = await api.get(`/review_tasks/${encodeURIComponent(taskId)}/results/download`, {
      responseType: 'blob'
    });
    return response.data;
  },

  async downloadManyResults(taskIds) {
    const response = await api.post('/review_tasks/results/download', { ids: taskIds }, {
      responseType: 'blob'
    });
    return response.data;
  },

  async deleteOne(taskId) {
    const response = await api.delete(`/review_tasks/${encodeURIComponent(taskId)}`);
    return response.data;
  },

  async deleteMany(taskIds) {
    const response = await api.delete('/review_tasks', { data: { ids: taskIds } });
    return response.data;
  }
};
