import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';
export const IMAGE_BASE = import.meta.env.VITE_IMAGE_BASE !== undefined ? import.meta.env.VITE_IMAGE_BASE : 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE
});

export default api;
