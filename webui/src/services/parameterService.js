import api from './api';

export const parameterService = {
  async list() {
    const response = await api.get('/parameters');
    return response.data;
  },

  async create(parameter) {
    const response = await api.post('/parameters', parameter);
    return response.data;
  },

  async update(key, parameter) {
    const response = await api.put(`/parameters/${encodeURIComponent(key)}`, parameter);
    return response.data;
  },

  async delete(key) {
    const response = await api.delete(`/parameters/${encodeURIComponent(key)}`);
    return response.data;
  }
};
