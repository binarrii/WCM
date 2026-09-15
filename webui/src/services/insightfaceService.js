import api from './api';

export const insightfaceService = {
  async sync(nodeId = null) {
    return (await api.post('/insightface/replication/sync', { node_id: nodeId }, { timeout: 15000 })).data;
  }
};
