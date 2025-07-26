const API_BASE_URL = 'http://localhost:8000';

export interface ModelStats {
  samplesProcessed: number;
  lastTrainingTime: string;
  modelVersion: number;
  accuracyHistory: number[];
  errorHistory: number[];
  bufferSize: number;
  isTraining: boolean;
}

export interface CollectionStats {
  totalSamples: number;
  successfulPredictions: number;
  failedPredictions: number;
  bufferSize: number;
  symbolsCount: number;
  isRunning: boolean;
}

export interface DebugStats {
  trade_processor: {
    processed_trades: Record<string, number>;
    wash_trade_alerts: Record<string, number>;
    ping_pong_alerts: Record<string, number>;
    ramping_alerts: Record<string, number>;
    trade_history_sizes: Record<string, number>;
    volume_cluster_sizes: Record<string, number>;
    kline_data_symbols: number;
    consecutive_long_counts: Record<string, number>;
    recent_alerts_count: Record<string, number>;
  };
  orderbook_analyzer: {
    analysis_counts: Record<string, number>;
    alert_counts: Record<string, number>;
    trade_history_sizes: Record<string, number>;
    orderbook_history_sizes: Record<string, number>;
    volume_profile_sizes: Record<string, number>;
    recent_alerts_count: Record<string, number>;
  };
  uptime_hours: number;
}

export interface Alert {
  id: number;
  symbol: string;
  alert_type: string;
  price: number;
  alert_timestamp_ms: number;
  message: string;
  volume_ratio?: number;
  current_volume_usdt?: number;
  average_volume_usdt?: number;
  consecutive_count?: number;
  grouped_alerts_count: number;
  is_grouped: boolean;
  group_id?: string;
  has_imbalance: boolean;
  imbalance_data?: any;
  candle_data?: any;
  order_book_snapshot?: any;
  trade_history?: any;
  status: string;
  is_true_signal?: boolean;
  predicted_price_change?: number;
  predicted_direction?: string;
  ml_source_alert_type?: string;
  created_at: string;
}

export interface AlertsResponse {
  alerts: Alert[];
  total: number;
  page: number;
  limit: number;
}

export interface PredictionResponse {
  symbol: string;
  predicted_price_change: number;
  predicted_direction: string;
  timestamp: string;
  model_status: any;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Получение статистики отладки (содержит информацию о ML системе)
  async getDebugStats(): Promise<DebugStats> {
    return this.request<DebugStats>('/debug/stats');
  }

  // Получение алертов
  async getAlerts(params?: {
    limit?: number;
    offset?: number;
    symbol?: string;
    alert_type?: string;
    status?: string;
  }): Promise<AlertsResponse> {
    const searchParams = new URLSearchParams();

    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.offset) searchParams.append('offset', params.offset.toString());
    if (params?.symbol) searchParams.append('symbol', params.symbol);
    if (params?.alert_type) searchParams.append('alert_type', params.alert_type);
    if (params?.status) searchParams.append('status', params.status);

    const query = searchParams.toString();
    return this.request<AlertsResponse>(`/alerts${query ? `?${query}` : ''}`);
  }

  // Получение предсказания для символа
  async getPrediction(symbol: string): Promise<PredictionResponse> {
    return this.request<PredictionResponse>(`/predict_price/${symbol}`);
  }

  // Получение статистики алертов
  async getAlertsStatistics(days: number = 7): Promise<any> {
    return this.request<any>(`/alerts/statistics?days=${days}`);
  }
}

export const apiClient = new ApiClient();