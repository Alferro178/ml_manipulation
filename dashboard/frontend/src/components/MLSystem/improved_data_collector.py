import asyncio
import logging
import json
import time
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
import aiofiles
import pickle

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Структура для хранения точки данных"""
    timestamp: float
    symbol: str
    features: Dict[str, float]
    target_price: Optional[float] = None
    target_direction: Optional[int] = None
    alert_data: Optional[Dict[str, Any]] = None

class MLDataCollector:
    """
    Улучшенный сборщик данных для ML с поддержкой:
    - Потокового сбора данных
    - Автоматической обработки данных разного масштаба
    - Персистентного хранения
    - Адаптивного сбора признаков
    """
    
    def __init__(self, db_manager, ml_model, feature_engineer,
                 data_collection_window_sec=30,
                 target_prediction_window_sec=300,
                 min_data_points=50,
                 max_buffer_size=10000,
                 persistence_file="ml_data_buffer.pkl"):
        
        self.db_manager = db_manager
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        self.data_collection_window_sec = data_collection_window_sec
        self.target_prediction_window_sec = target_prediction_window_sec
        self.min_data_points = min_data_points
        self.max_buffer_size = max_buffer_size
        self.persistence_file = persistence_file
        
        # Буферы для данных
        self.data_buffer = deque(maxlen=max_buffer_size)
        self.processed_alerts = defaultdict(float)  # Cooldown tracking
        self.symbol_data = defaultdict(lambda: {
            'recent_prices': deque(maxlen=1000),
            'recent_volumes': deque(maxlen=1000),
            'recent_features': deque(maxlen=100),
            'last_update': 0
        })
        
        # Статистика сбора данных
        self.collection_stats = {
            'total_samples': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'last_collection_time': None,
            'symbols_tracked': set(),
            'feature_importance': defaultdict(float)
        }
        
        # Потокобезопасность
        self.buffer_lock = threading.RLock()
        self.stats_lock = threading.Lock()
        
        # Асинхронные задачи
        self.collection_task = None
        self.persistence_task = None
        self.is_running = False
        
        # Загрузка сохраненных данных
        self._load_persistent_data()
        
        logger.info("ImprovedMLDataCollector initialized successfully.")

    async def start_collection(self):
        """Запуск процесса сбора данных"""
        if self.is_running:
            logger.warning("Data collection already running")
            return
        
        self.is_running = True
        
        # Запуск асинхронных задач
        self.collection_task = asyncio.create_task(self._continuous_collection())
        self.persistence_task = asyncio.create_task(self._periodic_persistence())
        
        logger.info("Data collection started")

    async def stop_collection(self):
        """Остановка процесса сбора данных"""
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
        
        if self.persistence_task:
            self.persistence_task.cancel()
        
        # Сохранение данных перед остановкой
        await self._save_persistent_data()
        
        logger.info("Data collection stopped")

    async def _continuous_collection(self):
        """Непрерывный сбор данных"""
        while self.is_running:
            try:
                await self._collect_market_data()
                await asyncio.sleep(self.data_collection_window_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(5)

    async def _collect_market_data(self):
        """Сбор рыночных данных для всех отслеживаемых символов"""
        try:
            current_time = time.time()
            
            # Получение списка активных символов
            active_symbols = await self._get_active_symbols()
            
            for symbol in active_symbols:
                try:
                    # Сбор данных для символа
                    data_point = await self._collect_symbol_data(symbol, current_time)
                    
                    if data_point:
                        with self.buffer_lock:
                            self.data_buffer.append(data_point)
                        
                        # Обновление статистики
                        with self.stats_lock:
                            self.collection_stats['total_samples'] += 1
                            self.collection_stats['symbols_tracked'].add(symbol)
                            self.collection_stats['last_collection_time'] = datetime.now().isoformat()
                
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error in market data collection: {e}")

    async def _collect_symbol_data(self, symbol: str, timestamp: float) -> Optional[DataPoint]:
        """Сбор данных для конкретного символа"""
        try:
            # Получение текущих рыночных данных
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return None
            
            # Извлечение признаков
            features = await self._extract_enhanced_features(symbol, market_data, timestamp)
            if not features:
                return None
            
            # Нормализация признаков для разных масштабов
            normalized_features = self._normalize_market_features(features, symbol)
            
            # Создание точки данных
            data_point = DataPoint(
                timestamp=timestamp,
                symbol=symbol,
                features=normalized_features
            )
            
            # Попытка определить целевые значения для обучения
            await self._set_target_values(data_point, symbol)
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return None

    def _normalize_market_features(self, features: Dict[str, float], symbol: str) -> Dict[str, float]:
        """Нормализация признаков для работы с данными разного масштаба"""
        try:
            normalized = {}
            
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    normalized[key] = 0.0
                    continue
                
                # Специальная обработка для разных типов признаков
                if 'price' in key.lower():
                    # Для цен используем логарифмическое масштабирование
                    if value > 0:
                        # Относительное изменение цены
                        recent_prices = self.symbol_data[symbol]['recent_prices']
                        if len(recent_prices) > 0:
                            ref_price = np.median(list(recent_prices))
                            normalized[key] = np.log1p(abs(value - ref_price) / ref_price) * np.sign(value - ref_price)
                        else:
                            normalized[key] = np.log1p(value)
                    else:
                        normalized[key] = 0.0
                        
                elif 'volume' in key.lower():
                    # Для объемов используем логарифмическое масштабирование
                    if value > 0:
                        normalized[key] = np.log1p(value)
                    else:
                        normalized[key] = 0.0
                        
                elif 'ratio' in key.lower() or 'spread' in key.lower():
                    # Для отношений и спредов используем ограничение
                    normalized[key] = np.clip(value, -10, 10)
                    
                elif 'count' in key.lower():
                    # Для счетчиков используем квадратный корень
                    normalized[key] = np.sqrt(max(0, value))
                    
                else:
                    # Для остальных признаков используем стандартное ограничение
                    normalized[key] = np.clip(value, -100, 100)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features

    async def _extract_enhanced_features(self, symbol: str, market_data: Dict, timestamp: float) -> Optional[Dict[str, float]]:
        """Извлечение расширенного набора признаков"""
        try:
            # Базовые признаки от FeatureEngineer
            base_features = {}
            if self.feature_engineer:
                order_book = market_data.get('order_book', {})
                trade_history = market_data.get('trades', [])
                
                base_features = self.feature_engineer.extract_features(
                    symbol, order_book, trade_history, timestamp
                ) or {}
            
            # Дополнительные признаки
            enhanced_features = {}
            
            # Признаки volatility
            recent_prices = self.symbol_data[symbol]['recent_prices']
            if len(recent_prices) > 10:
                prices_array = np.array(list(recent_prices))
                enhanced_features['price_volatility'] = np.std(prices_array)
                enhanced_features['price_range'] = np.max(prices_array) - np.min(prices_array)
                enhanced_features['price_momentum'] = (prices_array[-1] - prices_array[0]) / prices_array[0] if prices_array[0] > 0 else 0
            
            # Признаки объема
            recent_volumes = self.symbol_data[symbol]['recent_volumes']
            if len(recent_volumes) > 10:
                volumes_array = np.array(list(recent_volumes))
                enhanced_features['volume_volatility'] = np.std(volumes_array)
                enhanced_features['volume_trend'] = np.polyfit(range(len(volumes_array)), volumes_array, 1)[0]
            
            # Временные признаки
            enhanced_features['hour_of_day'] = datetime.fromtimestamp(timestamp).hour
            enhanced_features['day_of_week'] = datetime.fromtimestamp(timestamp).weekday()
            
            # Технические индикаторы
            if len(recent_prices) > 20:
                prices_array = np.array(list(recent_prices))
                # Simple Moving Average
                enhanced_features['sma_ratio'] = prices_array[-1] / np.mean(prices_array[-20:]) if np.mean(prices_array[-20:]) > 0 else 1
                # RSI approximation
                price_changes = np.diff(prices_array[-14:])
                gains = price_changes[price_changes > 0]
                losses = abs(price_changes[price_changes < 0])
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                rs = avg_gain / avg_loss
                enhanced_features['rsi'] = 100 - (100 / (1 + rs))
            
            # Объединение всех признаков
            all_features = {**base_features, **enhanced_features}
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return None

    async def _set_target_values(self, data_point: DataPoint, symbol: str):
        """Установка целевых значений для обучения"""
        try:
            # Получение будущих цен для определения целевых значений
            future_price = await self._get_future_price(symbol, 
                                                      data_point.timestamp + self.target_prediction_window_sec)
            
            if future_price is not None:
                current_price = data_point.features.get('mid_price', 0)
                if current_price > 0:
                    price_change = (future_price - current_price) / current_price
                    data_point.target_price = price_change
                    
                    # Определение направления
                    if price_change > 0.001:  # >0.1%
                        data_point.target_direction = 1  # up
                    elif price_change < -0.001:  # <-0.1%
                        data_point.target_direction = -1  # down
                    else:
                        data_point.target_direction = 0  # neutral
            
        except Exception as e:
            logger.error(f"Error setting target values: {e}")

    async def add_alert_for_ml_processing(self, alert_data: Dict[str, Any]):
        """Добавление алерта для ML обработки"""
        try:
            symbol = alert_data.get('symbol')
            alert_type = alert_data.get('alert_type')
            alert_id = alert_data.get('id')
            
            if not all([symbol, alert_type, alert_id is not None]):
                logger.warning("Incomplete alert data for ML processing")
                return
            
            alert_key = f"{symbol}_{alert_type}_{alert_id}"
            current_time = time.time()
            
            # Проверка cooldown
            if current_time - self.processed_alerts.get(alert_key, 0) < 30:  # 30 sec cooldown
                return
            
            # Обработка алерта
            await self._process_alert_for_ml(alert_data)
            
            # Обновление cooldown
            self.processed_alerts[alert_key] = current_time
            
        except Exception as e:
            logger.error(f"Error processing alert for ML: {e}")

    async def _process_alert_for_ml(self, alert_data: Dict[str, Any]):
        """Обработка алерта для машинного обучения"""
        try:
            symbol = alert_data['symbol']
            timestamp = alert_data.get('alert_timestamp_ms', time.time() * 1000) / 1000
            
            # Сбор данных в момент алерта
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return
            
            # Извлечение признаков
            features = await self._extract_enhanced_features(symbol, market_data, timestamp)
            if not features:
                return
            
            # Нормализация
            normalized_features = self._normalize_market_features(features, symbol)
            
            # Создание точки данных
            data_point = DataPoint(
                timestamp=timestamp,
                symbol=symbol,
                features=normalized_features,
                alert_data=alert_data
            )
            
            # Установка целевых значений
            await self._set_target_values(data_point, symbol)
            
            # Добавление в буфер
            with self.buffer_lock:
                self.data_buffer.append(data_point)
            
            # Если есть целевые значения, добавляем в модель для обучения
            if data_point.target_price is not None and data_point.target_direction is not None:
                self.ml_model.add_training_data(
                    data_point.features,
                    data_point.target_price,
                    data_point.target_direction
                )
                
                with self.stats_lock:
                    self.collection_stats['successful_predictions'] += 1
            
        except Exception as e:
            logger.error(f"Error processing alert for ML: {e}")
            with self.stats_lock:
                self.collection_stats['failed_predictions'] += 1

    async def _get_active_symbols(self) -> List[str]:
        """Получение списка активных символов для отслеживания"""
        try:
            # Здесь должна быть логика получения активных символов
            # Пока используем стандартный набор
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return []

    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Получение рыночных данных для символа"""
        try:
            # Здесь должна быть интеграция с реальным источником данных
            # Пока возвращаем заглушку
            return {
                'order_book': {'bids': [], 'asks': []},
                'trades': [],
                'ticker': {}
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _get_future_price(self, symbol: str, future_timestamp: float) -> Optional[float]:
        """Получение будущей цены для определения целевых значений"""
        try:
            # Здесь должна быть логика получения исторических данных
            # для определения цены в будущем моменте времени
            return None
        except Exception as e:
            logger.error(f"Error getting future price for {symbol}: {e}")
            return None

    async def _periodic_persistence(self):
        """Периодическое сохранение данных"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Сохранение каждые 5 минут
                await self._save_persistent_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic persistence: {e}")

    async def _save_persistent_data(self):
        """Сохранение данных в файл"""
        try:
            data_to_save = {
                'data_buffer': list(self.data_buffer),
                'collection_stats': self.collection_stats,
                'symbol_data': dict(self.symbol_data),
                'processed_alerts': dict(self.processed_alerts)
            }
            
            async with aiofiles.open(self.persistence_file, 'wb') as f:
                await f.write(pickle.dumps(data_to_save))
                
            logger.debug(f"Data saved to {self.persistence_file}")
            
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")

    def _load_persistent_data(self):
        """Загрузка сохраненных данных"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Восстановление данных
                saved_buffer = data.get('data_buffer', [])
                for item in saved_buffer[-1000:]:  # Загружаем последние 1000 записей
                    self.data_buffer.append(item)
                
                self.collection_stats.update(data.get('collection_stats', {}))
                
                # Конвертация обратно в defaultdict
                symbol_data = data.get('symbol_data', {})
                for symbol, sym_data in symbol_data.items():
                    self.symbol_data[symbol].update(sym_data)
                
                self.processed_alerts.update(data.get('processed_alerts', {}))
                
                logger.info(f"Loaded {len(self.data_buffer)} data points from persistence")
                
        except Exception as e:
            logger.warning(f"Could not load persistent data: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Получение статистики сбора данных"""
        with self.stats_lock:
            return {
                **self.collection_stats,
                'buffer_size': len(self.data_buffer),
                'symbols_count': len(self.symbol_data),
                'is_running': self.is_running
            }

    async def cleanup(self):
        """Очистка ресурсов"""
        await self.stop_collection()
        self.data_buffer.clear()
        self.symbol_data.clear()
        logger.info("MLDataCollector cleanup completed")