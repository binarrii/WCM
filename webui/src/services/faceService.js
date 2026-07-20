import api from './api';

export const faceService = {
  // Get face records with pagination, search, and type filters
  async getRecords({ page = 1, limit = 12, search = '', type = 'All' }) {
    const response = await api.get('/face_records', {
      params: { page, limit, search, type }
    });
    return response.data;
  },

  // Get statistics/counts for categories
  async getStats() {
    const response = await api.get('/face_records/stats');
    return response.data;
  },

  // Create face record (FormData)
  async createRecord(formData) {
    const response = await api.post('/face_records', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // Update face record
  async updateRecord(id, data) {
    const response = await api.put(`/face_records/${id}`, data);
    return response.data;
  },

  // Delete face record
  async deleteRecord(id) {
    const response = await api.delete(`/face_records/${id}`);
    return response.data;
  },

  // Search faces by image file (FormData) or JSON (url)
  async searchFaces(payload) {
    if (payload instanceof FormData) {
      const response = await api.post('/search', payload, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } else {
      const response = await api.post('/search', payload);
      return response.data;
    }
  }
};
