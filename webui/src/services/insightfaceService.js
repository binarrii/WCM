import api from './api';

export const insightfaceService = {
  async status() {
    return (await api.get('/insightface/replication', { timeout: 15000 })).data;
  },
  async sync(nodeId = null) {
    return (await api.post('/insightface/replication/sync', { node_id: nodeId }, { timeout: 15000 })).data;
  }
};
