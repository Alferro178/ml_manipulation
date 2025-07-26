import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',  // Измените на ваш порт FastAPI, если другой
  timeout: 10000,
});

api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error);
    throw error;
  }
);

export const apiClient = {
  async getDebugStats(): Promise<any> {
    const response = await api.get('/debug/stats');
    return response.data;
  },

  async getAlerts(params: { limit: number }): Promise<{ alerts: any[] }> {
    const response = await api.get('/alerts', { params });
    return response.data;
  },

  async getPrediction(symbol: string): Promise<any> {
    const response = await api.get(`/predict_price/${symbol}`);
    return response.data;
  },
};