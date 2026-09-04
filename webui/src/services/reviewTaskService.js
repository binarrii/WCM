import api from './api';

export const reviewTaskService = {
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

  async deleteOne(taskId) {
    const response = await api.delete(`/review_tasks/${encodeURIComponent(taskId)}`);
    return response.data;
  },

  async deleteMany(taskIds) {
    const response = await api.delete('/review_tasks', { data: { ids: taskIds } });
    return response.data;
  }
};
