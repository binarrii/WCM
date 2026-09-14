import api from './api';

const mutationHeaders = revision => ({
  'Idempotency-Key': Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join(''),
  ...(revision ? { 'If-Match': revision } : {}),
});

export const faceService = {
  async findSameName(name) {
    const response = await api.get('/face_records/name_matches', { params: { name } });
    return response.data.items;
  },
  // Get face records with pagination, search, and type filters
  async getRecords({ cursor = null, limit = 12, search = '', type = 'All' }) {
    const response = await api.get('/face_records', {
      params: { cursor, limit, search, type }
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
        ...mutationHeaders(),
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // Update face record
  async appendImage(id, formData, revision) {
    const response = await api.post(`/face_records/${encodeURIComponent(id)}/images`, formData, { headers: mutationHeaders(revision) });
    return response.data;
  },

  async deleteImages(id, imageUrls, revision) {
    const response = await api.delete(`/face_records/${encodeURIComponent(id)}/images`, {
      data: { image_urls: imageUrls }, headers: mutationHeaders(revision)
    });
    return response.data;
  },

  async mergeRecords(targetId, sourceIds, revision) {
    const response = await api.post('/face_records/merge', { target_id: targetId, source_ids: sourceIds }, { headers: mutationHeaders(revision) });
    return response.data;
  },

  async updateRecord(id, data, revision) {
    const response = await api.put(`/face_records/${id}`, data, { headers: mutationHeaders(revision) });
    return response.data;
  },

  // Delete face record
  async deleteRecord(id, revision) {
    const response = await api.delete(`/face_records/${id}`, { headers: mutationHeaders(revision) });
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
