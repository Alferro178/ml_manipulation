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
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Структура для хранения точки данных"""
    timestamp: float
    symbol: str
    features: Dict[str, float]
    target_price: Optional[float] = None
    target_direction: Optional[str] = None
    alert_data: Optional[Dict[str, Any]] = None


class MLDataCollector:
    def __init__(self, db_manager, ml_model, feature_engineer,
                 trade_processor=None, orderbook_analyzer=None,
                 data_collection_window_sec=30,
                 target_prediction_window_sec=300,
                 min_data_points=5,  # Снижаем минимальный порог
                 max_buffer_size=10000,
                 persistence_file="ml_data_buffer.pkl"):

        self.db_manager = db_manager
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        self.trade_processor = trade_processor
        self.orderbook_analyzer = orderbook_analyzer
        self.data_collection_window_sec = data_collection_window_sec
        self.target_prediction_window_sec = target_prediction_window_sec
        self.min_data_points = min_data_points
        self.max_buffer_size = max_buffer_size
        self.persistence_file = persistence_file

        self.data_buffer = deque(maxlen=max_buffer_size)
        self.processed_alerts = defaultdict(float)
        self.symbol_data = defaultdict(lambda: {
            'recent_prices': deque(maxlen=1000),
            'recent_volumes': deque(maxlen=1000),
            'recent_features': deque(maxlen=100),
            'last_update': 0
        })

        self.pending_alerts = asyncio.Queue()

        self.collection_stats = {
            'total_samples': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'last_collection_time': None,
            'symbols_tracked': set(),
            'feature_importance': defaultdict(float)
        }

        self.buffer_lock = threading.RLock()
        self.stats_lock = threading.Lock()

        self.collection_task = None
        self.persistence_task = None
        self.training_task = None
        self.is_running = False

        self.http_session = None

        self._load_persistent_data()

        logger.info("ImprovedMLDataCollector initialized successfully.")

    async def start_collection(self):
        """Запуск процесса сбора данных"""
        if self.is_running:
            logger.warning("Data collection already running")
            return

        self.is_running = True
        # Создаем HTTP-сессию с таймаутом
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.http_session = aiohttp.ClientSession(timeout=timeout)
        self.http_session = aiohttp.ClientSession()

        self.collection_task = asyncio.create_task(self._continuous_collection())
        self.persistence_task = asyncio.create_task(self._periodic_persistence())
        # Запускаем обучение чаще для тестирования
        self.training_task = asyncio.create_task(self.train_model_periodically(interval_hours=0.05))  # Каждые 3 минуты

        # Добавляем задачу обработки ожидающих алертов
        self.alert_processing_task = asyncio.create_task(self.process_pending_alerts_loop(5))

        logger.info("Data collection and training started")

    async def stop_collection(self):
        """Остановка процесса сбора данных"""
        self.is_running = False

        if self.collection_task:
            self.collection_task.cancel()
        if self.persistence_task:
            self.persistence_task.cancel()
        if self.training_task:
            self.training_task.cancel()
        if hasattr(self, 'alert_processing_task') and self.alert_processing_task:
            self.alert_processing_task.cancel()

        if self.http_session:
            await self.http_session.close()

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
            active_symbols = await self._get_active_symbols()

            for symbol in active_symbols:
                try:
                    data_point = await self._collect_symbol_data(symbol, current_time)

                    if data_point:
                        with self.buffer_lock:
                            self.data_buffer.append(data_point)

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
            order_book_snapshot = self.orderbook_analyzer.current_orderbooks.get(
                symbol) if self.orderbook_analyzer else None
            trade_history = list(
                self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else []

            if not order_book_snapshot:
                logger.debug(f"No order book for {symbol}")
                return None

            features = self.feature_engineer.extract_features(symbol, order_book_snapshot, trade_history, timestamp)
            if not features:
                logger.debug(f"No features extracted for {symbol}")
                return None

            normalized_features = self._normalize_market_features(features, symbol)

            data_point = DataPoint(
                timestamp=timestamp,
                symbol=symbol,
                features=normalized_features
            )

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
                    logger.warning(f"Invalid feature value for {key} in {symbol}: {value}, setting to 0.0")
                    normalized[key] = 0.0
                    continue

                if 'price' in key.lower():
                    if value > 0:
                        recent_prices = self.symbol_data[symbol]['recent_prices']
                        if len(recent_prices) > 0:
                            ref_price = np.median(list(recent_prices))
                            normalized[key] = np.log1p(abs(value - ref_price) / ref_price) * np.sign(value - ref_price)
                        else:
                            normalized[key] = np.log1p(value)
                    else:
                        normalized[key] = 0.0

                elif 'volume' in key.lower():
                    if value > 0:
                        normalized[key] = np.log1p(value)
                    else:
                        normalized[key] = 0.0

                elif 'ratio' in key.lower() or 'spread' in key.lower():
                    normalized[key] = np.clip(value, -10, 10)

                elif 'count' in key.lower():
                    normalized[key] = np.sqrt(max(0, value))

                else:
                    normalized[key] = np.clip(value, -100, 100)

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing features for {symbol}: {e}")
            return features

    async def add_alert_for_ml_processing(self, alert_data: Dict[str, Any]):
        """Добавление сообщения для обработки ML"""
        try:
            symbol = alert_data.get('symbol')
            alert_type = alert_data.get('alert_type')
            alert_id = alert_data.get('id')

            if not all([symbol, alert_type, alert_id is not None]):
                logger.warning("Incomplete alert data for ML processing")
                return

            alert_key = f"{symbol}_{alert_type}_{alert_id}"
            current_time = time.time()

            if current_time - self.processed_alerts.get(alert_key, 0) < 30:
                return

            await self.pending_alerts.put(alert_data)
            self.processed_alerts[alert_key] = current_time

            # Немедленно обрабатываем алерт, если очередь не слишком большая
            if self.pending_alerts.qsize() < 10:
                try:
                    alert_to_process = await asyncio.wait_for(self.pending_alerts.get(), timeout=0.1)
                    await self._process_alert_for_ml(alert_to_process)
                except asyncio.TimeoutError:
                    pass  # Очередь пуста, это нормально

        except Exception as e:
            logger.error(f"Error adding alert for ML: {e}")

    async def _process_alert_for_ml(self, alert_data: Dict[str, Any]):
        """Обработка сообщения для машинного обучения"""
        try:
            symbol = alert_data['symbol']
            alert_id = alert_data.get('id')
            timestamp = alert_data.get('alert_timestamp_ms', time.time() * 1000) / 1000

            order_book_snapshot = alert_data.get('order_book_snapshot')
            trade_history = alert_data.get('trade_history', [])

            logger.debug(
                f"Alert {alert_id} for {symbol}: snapshot={bool(order_book_snapshot)}, trades={len(trade_history)}")

            if not order_book_snapshot:
                logger.warning(f"No order book snapshot in alert_data for {symbol}, alert {alert_id}")
                return

            if self.trade_processor and not trade_history:
                trade_history_in_processor = self.trade_processor.trade_history.get(symbol)
                if trade_history_in_processor is None:
                    logger.warning(
                        f"No trade history data in trade_processor for {symbol}. Fetching historical trades.")
                    trade_history = await self._fetch_historical_trades(symbol, timestamp)
                else:
                    trade_history = list(trade_history_in_processor)
                logger.debug(f"TradeProcessor trade_history for {symbol}: {len(trade_history)} trades")

            if not trade_history:
                trade_history = await self._fetch_historical_trades(symbol, timestamp)
                logger.debug(f"Fetched {len(trade_history)} historical trades for {symbol}")

            features = self.feature_engineer.extract_features(symbol, order_book_snapshot, trade_history, timestamp)
            if not features:
                logger.warning(f"No features extracted for {symbol}, alert {alert_id}")
                return

            normalized_features = self._normalize_market_features(features, symbol)
            if not normalized_features:
                logger.warning(f"No normalized features for {symbol}, alert {alert_id}, skipping data point")
                return

            data_point = DataPoint(
                timestamp=timestamp,
                symbol=symbol,
                features=normalized_features,
                alert_data=alert_data
            )

            await self._set_target_values(data_point, symbol)
            logger.debug(
                f"Alert {alert_id}: target_price={data_point.target_price}, direction={data_point.target_direction}")

            if data_point.target_price is None or data_point.target_price == 0.0:
                logger.warning(f"Skipping data point for {symbol}, alert {alert_id} due to invalid target_price")
                return

            with self.buffer_lock:
                self.data_buffer.append(data_point)
                logger.debug(f"Added data point to buffer for {symbol}, buffer size: {len(self.data_buffer)}")

            target_price = data_point.target_price
            target_direction = data_point.target_direction

            # Сохраняем данные в базу с обработкой ошибок
            try:
                await self.db_manager.insert_ml_training_data(
                    symbol,
                    normalized_features,
                    target_price,
                    target_direction,
                    alert_data['id']
                )
                logger.debug(f"ML training data inserted for {symbol}, alert {alert_id}")
            except Exception as db_error:
                logger.error(f"Failed to insert ML training data for {symbol}, alert {alert_id}: {db_error}")
                # Продолжаем работу даже если не удалось сохранить в БД

            # Проверяем, нужно ли запустить обучение
            buffer_size = len(self.data_buffer)
            if buffer_size >= self.min_data_points and buffer_size % 5 == 0:  # Каждые 5 новых точек данных
                logger.info(f"Buffer reached {buffer_size} points, considering training")
                # Запускаем обучение в фоне, если прошло достаточно времени
                last_training = self.collection_stats.get('last_training_time')
                if not last_training or (
                        time.time() - time.mktime(datetime.fromisoformat(last_training).timetuple())) > 180:  # 3 минуты
                    asyncio.create_task(self._trigger_immediate_training())

            with self.stats_lock:
                self.collection_stats['successful_predictions'] += 1

        except Exception as e:
            logger.error(f"Error processing alert for ML: {e}", exc_info=True)
            with self.stats_lock:
                self.collection_stats['failed_predictions'] += 1

    async def _trigger_immediate_training(self):
        """Запуск немедленного обучения"""
        try:
            logger.info("Triggering immediate training due to sufficient data accumulation")

            # Сначала пытаемся использовать данные из буфера
            buffer_data = []
            with self.buffer_lock:
                buffer_data = list(self.data_buffer)

            if len(buffer_data) >= self.min_data_points:
                logger.info(f"Using {len(buffer_data)} data points from buffer for immediate training")

                features = []
                target_prices = []
                target_directions = []

                for data_point in buffer_data:
                    if (data_point.features and data_point.target_price is not None and
                            data_point.target_direction is not None):
                        features.append(data_point.features)
                        target_prices.append(data_point.target_price)
                        target_directions.append(data_point.target_direction)

                if len(features) >= self.min_data_points:
                    await self._train_with_data(features, target_prices, target_directions, "buffer")
                    return

            # Если данных в буфере недостаточно, пытаемся получить из БД
            training_data = await self.db_manager.get_ml_training_data(limit=1000)

            if len(training_data) >= self.min_data_points:  # Минимум для обучения
                features = []
                target_prices = []
                target_directions = []

                for row in training_data:
                    try:
                        feature_dict = json.loads(row['features'])
                        target_price = row['target_price_change']
                        target_direction = row['target_direction']

                        if (feature_dict and isinstance(feature_dict, dict) and
                                all(isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)) for v in
                                    feature_dict.values()) and
                                isinstance(target_price, (int, float)) and not (
                                        np.isnan(target_price) or np.isinf(target_price)) and
                                target_direction in ['up', 'down', 'neutral']):
                            features.append(feature_dict)
                            target_prices.append(target_price)
                            target_directions.append(target_direction)
                    except:
                        continue

                if len(features) >= self.min_data_points:
                    await self._train_with_data(features, target_prices, target_directions, "database")
            else:
                logger.warning(f"Insufficient training data: buffer={len(buffer_data)}, db={len(training_data)}")

        except Exception as e:
            logger.error(f"Error in immediate training: {e}")

    async def _train_with_data(self, features, target_prices, target_directions, source):
        """Обучение модели с предоставленными данными"""
        try:
            logger.info(f"Training model with {len(features)} samples from {source}")

            # Подготовка данных
            if isinstance(features[0], dict):
                feature_names = sorted(features[0].keys())
                X = pd.DataFrame(features, columns=feature_names).fillna(0).values
            else:
                # Если features уже в виде массива
                X = np.array(features)
                feature_names = self.ml_model.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

            y_price = np.array(target_prices)
            y_direction = np.array([1 if d == 'up' else -1 if d == 'down' else 0 for d in target_directions])

            # Проверка данных
            if np.all(y_price == 0) or np.std(y_price) < 1e-6:
                logger.warning("All target prices are zero or have no variance, adding noise")
                noise = np.random.normal(0, 0.01, len(y_price))
                y_price = y_price + noise

            # Обучение модели
            self.ml_model.feature_names = feature_names
            metrics = self.ml_model.train(X, y_price, y_direction)
            self.ml_model.save_models()

            self.collection_stats['last_training_time'] = datetime.now().isoformat()
            logger.info(f"Training completed with {len(features)} samples from {source}. Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error in training with data from {source}: {e}")

    async def _fetch_historical_trades(self, symbol: str, timestamp: float) -> List[Dict]:
        """Загрузка исторических сделок через Bybit API с повторными попытками"""
        if not self.http_session:
            logger.error("HTTP session not initialized")
            return []

        max_attempts = 5
        retry_delay = 10
        attempt = 0

        # Проверка активности символа
        try:
            async with self.http_session.get(
                    f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}") as response:  # Изменено на linear
                if response.status != 200:
                    logger.error(f"Symbol {symbol} may not be active: HTTP {response.status}")
                    return []
                try:
                    data = await response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response for {symbol}: {e}")
                    return []
                if data is None:
                    logger.warning(f"Invalid response format for {symbol}: {data}")
                    return []
                if not isinstance(data, dict):
                    logger.warning(f"Response is not a dictionary for {symbol}: {type(data)}")
                    return []
                if data.get('retCode') != 0:
                    logger.warning(f"API error for {symbol}: {data.get('retMsg', 'Unknown error')}")
                    return []
                result = data.get('result')
                if not isinstance(result, dict) or not result.get('list'):
                    logger.warning(f"No ticker data for {symbol}, possibly invalid or inactive symbol")
                    return []
        except Exception as e:
            logger.error(f"Error checking symbol {symbol} activity: {e}")
            return []

        while attempt < max_attempts:
            try:
                end_time = int(timestamp * 1000)
                start_time = end_time - (3600 * 1000)  # Последний час данных
                url = f"https://api.bybit.com/v5/market/recent-trade?category=linear&symbol={symbol}&limit=1000"  # Изменено на linear

                # Используем правильные параметры для API
                params = {}
                async with self.http_session.get(url, params=params) as response:
                    if response.status == 429:
                        logger.warning(f"Rate limit exceeded for {symbol}, retrying after {retry_delay} seconds")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    if response.status != 200:
                        logger.error(f"Failed to fetch historical trades for {symbol}: HTTP {response.status}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue

                    try:
                        data = await response.json()
                    except Exception as e:
                        logger.error(f"Failed to parse JSON response for trades in {symbol}: {e}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue

                    if data is None:
                        logger.warning(f"Invalid response format for trades in {symbol}: {data}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    if not isinstance(data, dict):
                        logger.warning(f"Response is not a dictionary for trades in {symbol}: {type(data)}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue

                    if data.get('retCode') != 0:
                        logger.warning(f"API error fetching trades for {symbol}: {data.get('retMsg', 'Unknown error')}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue

                    result = data.get('result')
                    if not isinstance(result, dict):
                        logger.warning(f"Invalid result format for trades in {symbol}: {result}")
                        attempt += 1
                        await asyncio.sleep(retry_delay)
                        continue

                    trades = result.get('list', [])

                    # Фильтруем сделки по времени
                    filtered_trades = []
                    for trade in trades:
                        try:
                            trade_time = float(trade['time']) / 1000
                            if start_time / 1000 <= trade_time <= end_time / 1000:
                                filtered_trades.append(trade)
                        except (KeyError, ValueError):
                            continue

                    formatted_trades = []
                    for trade in filtered_trades:
                        try:
                            formatted_trades.append({
                                'timestamp': float(trade['time']) / 1000,
                                'price': float(trade['price']),
                                'size': float(trade['qty']),
                                'side': trade['side'],
                                'volume_usdt': float(trade['price']) * float(trade['qty']),
                                'trade_id': trade['tradeId']
                            })
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Invalid trade data for {symbol}: {e}")
                            continue

                    logger.debug(f"Fetched {len(formatted_trades)} historical trades for {symbol}")
                    return formatted_trades

            except Exception as e:
                logger.error(f"Error fetching historical trades for {symbol} (attempt {attempt + 1}): {e}")
                attempt += 1
                await asyncio.sleep(retry_delay)

        logger.error(f"Failed to fetch historical trades for {symbol} after {max_attempts} attempts")
        return []

    async def _get_active_symbols(self) -> List[str]:
        """Получение списка активных символов для отслеживания"""
        try:
            symbols = await self.db_manager.get_watchlist_symbols()
            logger.debug(f"Retrieved {len(symbols)} active symbols from watchlist")
            return symbols
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return ['BTCUSDT']

    async def _set_target_values(self, data_point: DataPoint, symbol: str):
        """Установка целевых значений для точки данных"""
        try:
            future_timestamp = data_point.timestamp + self.target_prediction_window_sec
            future_price = await self._get_future_price(symbol, future_timestamp)
            current_price = self._get_current_price(symbol)

            logger.debug(
                f"Setting target values for {symbol}: future_price={future_price}, current_price={current_price}")

            if future_price is None or current_price is None:
                logger.warning(
                    f"Cannot set target values for {symbol}: future_price={future_price}, current_price={current_price}")
                data_point.target_price = None
                data_point.target_direction = None
                return

            if current_price == 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                data_point.target_price = None
                data_point.target_direction = None
                return

            target_price = (future_price - current_price) / current_price * 100

            # Проверяем, что target_price не равен 0 или очень мал
            if abs(target_price) < 0.001:  # Менее 0.001%
                # Добавляем небольшой шум для избежания нулевых целевых значений
                noise = np.random.normal(0, 0.01)  # Шум в пределах 0.01%
                target_price += noise
                logger.debug(f"Added noise to target_price for {symbol}: {target_price}")

            data_point.target_price = target_price

            # Более чувствительные пороги для направления
            if abs(target_price) < 0.005:  # Порог для нейтрального направления (0.005%)
                data_point.target_direction = 'neutral'
            elif target_price > 0.005:
                data_point.target_direction = 'up'
            else:
                data_point.target_direction = 'down'

            logger.debug(
                f"Set target values for {symbol}: target_price={target_price}, direction={data_point.target_direction}")

        except Exception as e:
            logger.error(f"Error setting target values for {symbol}: {e}", exc_info=True)
            data_point.target_price = None
            data_point.target_direction = None

    async def _get_future_price(self, symbol: str, future_timestamp: float) -> Optional[float]:
        """Получение будущей цены на основе исторических данных"""
        try:
            current_time = time.time()

            # Если будущее время еще не наступило, используем прогнозирование
            if future_timestamp > current_time:
                # Получаем недавние цены для прогнозирования тренда
                recent_prices = []

                if self.trade_processor and symbol in self.trade_processor.trade_history:
                    history = list(self.trade_processor.trade_history.get(symbol, deque()))
                    recent_trades = [t for t in history if current_time - t['timestamp'] <= 300]  # Последние 5 минут
                    recent_prices = [t['price'] for t in recent_trades[-10:]]  # Последние 10 сделок

                if len(recent_prices) >= 3:
                    # Простое прогнозирование на основе тренда
                    price_changes = [recent_prices[i] - recent_prices[i - 1] for i in range(1, len(recent_prices))]
                    avg_change = np.mean(price_changes)
                    time_diff = future_timestamp - current_time

                    # Прогнозируем изменение цены
                    predicted_change = avg_change * (time_diff / 60)  # Изменение за минуту
                    future_price = recent_prices[-1] + predicted_change

                    # Добавляем случайную волатильность
                    volatility = np.std(price_changes) if len(price_changes) > 1 else 0
                    noise = np.random.normal(0, volatility * 0.1)
                    future_price += noise

                    logger.debug(f"Predicted future price for {symbol}: {future_price} (trend-based)")
                    return future_price

                # Если недостаточно данных, используем текущую цену с небольшим изменением
                current_price = self._get_current_price(symbol)
                if current_price:
                    # Добавляем случайное изменение в пределах 0.1%
                    random_change = np.random.normal(0, current_price * 0.001)
                    future_price = current_price + random_change
                    logger.debug(f"Using current price with noise for {symbol}: {future_price}")
                    return future_price

            # Для исторических данных используем существующую логику
            if self.trade_processor and symbol in self.trade_processor.trade_history:
                history = list(self.trade_processor.trade_history.get(symbol, deque()))
                future_trades = [trade for trade in history if abs(trade['timestamp'] - future_timestamp) <= 30]
                if future_trades:
                    logger.debug(f"Found {len(future_trades)} future trades in trade_history for {symbol}")
                    return np.mean([trade['price'] for trade in future_trades])

            historical_trades = await self._fetch_historical_trades(symbol, future_timestamp)
            future_trades = [trade for trade in historical_trades if abs(trade['timestamp'] - future_timestamp) <= 30]
            if future_trades:
                logger.debug(f"Found {len(future_trades)} future trades in historical data for {symbol}")
                return np.mean([trade['price'] for trade in future_trades])

            # Попробовать получить цену через API тикера
            try:
                async with self.http_session.get(
                        f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and data.get('retCode') == 0 and data.get('result', {}).get('list'):
                            price = float(data['result']['list'][0]['lastPrice'])
                            logger.debug(f"Using API ticker price for {symbol}: {price}")
                            return price
            except Exception as e:
                logger.warning(f"Failed to fetch ticker price for {symbol}: {e}")

            if self.orderbook_analyzer and symbol in self.orderbook_analyzer.current_orderbooks:
                ob = self.orderbook_analyzer.current_orderbooks[symbol]
                if ob['bids'] and ob['asks']:
                    best_bid = max(ob['bids'].keys()) if isinstance(ob['bids'], dict) else ob['bids'][0][0]
                    best_ask = min(ob['asks'].keys()) if isinstance(ob['asks'], dict) else ob['asks'][0][0]
                    mid_price = (best_bid + best_ask) / 2
                    logger.debug(f"Using orderbook mid-price for {symbol}: {mid_price}")
                    return mid_price

            logger.warning(f"No future price data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error getting future price for {symbol}: {e}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены"""
        try:
            if self.trade_processor and symbol in self.trade_processor.trade_history:
                history = list(self.trade_processor.trade_history.get(symbol, deque()))
                if history:
                    return history[-1]['price']
            if self.orderbook_analyzer and symbol in self.orderbook_analyzer.current_orderbooks:
                ob = self.orderbook_analyzer.current_orderbooks[symbol]
                if ob['bids'] and ob['asks']:
                    # Правильно обрабатываем структуру orderbook
                    if isinstance(ob['bids'], dict) and isinstance(ob['asks'], dict):
                        best_bid = max(ob['bids'].keys()) if ob['bids'] else 0
                        best_ask = min(ob['asks'].keys()) if ob['asks'] else 0
                    else:
                        best_bid = ob['bids'][0][0] if ob['bids'] else 0
                        best_ask = ob['asks'][0][0] if ob['asks'] else 0

                    if best_bid > 0 and best_ask > 0:
                        return (best_bid + best_ask) / 2
            logger.warning(f"No current price for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def _periodic_persistence(self):
        """Периодическое сохранение данных"""
        while self.is_running:
            try:
                await asyncio.sleep(300)
                await self._save_persistent_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic persistence: {e}")

    async def _save_persistent_data(self):
        """Сохранение данных в файл"""
        try:
            with self.buffer_lock:
                buffer_size = len(self.data_buffer)
                data_to_save = {
                    'data_buffer': list(self.data_buffer),
                    'collection_stats': self.collection_stats,
                    'symbol_data': dict(self.symbol_data),
                    'processed_alerts': dict(self.processed_alerts)
                }

            logger.debug(f"Attempting to save {buffer_size} data points to {self.persistence_file}")

            directory = os.path.dirname(self.persistence_file) or '.'
            if not os.access(directory, os.W_OK):
                logger.error(f"No write permission for directory {directory}")
                return

            os.makedirs(directory, exist_ok=True)

            async with aiofiles.open(self.persistence_file, 'wb') as f:
                await f.write(pickle.dumps(data_to_save))

            logger.info(f"Successfully saved {buffer_size} data points to {self.persistence_file}")

        except Exception as e:
            logger.error(f"Error saving persistent data to {self.persistence_file}: {e}")

    def _load_persistent_data(self):
        """Загрузка сохраненных данных"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'rb') as f:
                    data = pickle.load(f)

                saved_buffer = data.get('data_buffer', [])
                for item in saved_buffer[-1000:]:
                    self.data_buffer.append(item)

                self.collection_stats.update(data.get('collection_stats', {}))

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

    async def process_pending_alerts_loop(self, interval_sec: int):
        """Цикл обработки ожидающих сообщений"""
        while self.is_running:
            try:
                while not self.pending_alerts.empty():
                    alert_data = await self.pending_alerts.get()
                    await self._process_alert_for_ml(alert_data)
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pending alerts loop: {e}")
                await asyncio.sleep(5)

    async def train_model_periodically(self, interval_hours: int):
        """Периодическое обучение модели"""
        while self.is_running:
            try:
                logger.debug("Starting model training cycle")

                # Сначала проверяем буфер
                buffer_data = []
                with self.buffer_lock:
                    buffer_data = list(self.data_buffer)

                logger.info(f"Buffer contains {len(buffer_data)} data points")

                # Если в буфере достаточно данных, используем их
                if len(buffer_data) >= self.min_data_points:
                    features = []
                    target_prices = []
                    target_directions = []

                    for data_point in buffer_data:
                        if (data_point.features and data_point.target_price is not None and
                                data_point.target_direction is not None):
                            features.append(data_point.features)
                            target_prices.append(data_point.target_price)
                            target_directions.append(data_point.target_direction)

                    if len(features) >= self.min_data_points:
                        logger.info(f"Training with {len(features)} samples from buffer")
                        await self._train_with_data(features, target_prices, target_directions, "periodic_buffer")
                        # Ждем до следующего цикла
                        await asyncio.sleep(min(interval_hours * 3600, 1800))
                        continue

                # Если данных в буфере недостаточно, пытаемся получить из БД
                training_data = await self.db_manager.get_ml_training_data(limit=10000)
                logger.info(f"Retrieved {len(training_data)} training data points")

                # Снижаем минимальный порог для начального обучения
                min_required = self.min_data_points  # Используем настроенный минимум
                if len(training_data) < min_required:
                    logger.warning(
                        f"Insufficient data for training: {len(training_data)} points, required: {min_required}")
                    await asyncio.sleep(interval_hours * 3600)
                    continue

                features = []
                target_prices = []
                target_directions = []
                valid_records = 0

                for row in training_data:
                    try:
                        feature_dict = json.loads(row['features'])
                        target_price = row['target_price_change']
                        target_direction = row['target_direction']

                        # Проверяем валидность данных
                        if (feature_dict and isinstance(feature_dict, dict) and
                                all(isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)) for v in
                                    feature_dict.values()) and
                                isinstance(target_price, (int, float)) and not (
                                        np.isnan(target_price) or np.isinf(target_price)) and
                                target_direction in ['up', 'down', 'neutral']):

                            features.append(feature_dict)
                            target_prices.append(target_price)
                            target_directions.append(target_direction)
                            valid_records += 1
                        else:
                            logger.debug(
                                f"Invalid data for record ID {row.get('id', 'unknown')} in {row.get('symbol', 'unknown')}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse features for record ID {row.get('id', 'unknown')}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing record: {e}")

                logger.info(f"Processed {valid_records} valid feature sets out of {len(training_data)} records")

                if valid_records < min_required:
                    logger.warning("No valid features found in training data")
                    await asyncio.sleep(interval_hours * 3600)
                    continue

                # Проверяем разнообразие целевых значений
                unique_directions = set(target_directions)
                logger.info(
                    f"Target direction distribution before augmentation: {dict(zip(*np.unique(target_directions, return_counts=True)))}")

                # Улучшаем разнообразие данных
                if len(unique_directions) < 3:  # Нужны все три класса
                    logger.warning(f"Insufficient diversity in target directions: {unique_directions}")

                    # Создаем синтетические данные для недостающих классов
                    missing_classes = {'up', 'down', 'neutral'} - unique_directions
                    for missing_class in missing_classes:
                        # Добавляем несколько синтетических примеров
                        for _ in range(3):
                            # Копируем случайный существующий пример
                            idx = np.random.randint(0, len(features))
                            new_features = features[idx].copy()

                            if missing_class == 'up':
                                new_target_price = abs(np.random.normal(0.05, 0.02))  # Положительное изменение
                                new_target_direction = 'up'
                            elif missing_class == 'down':
                                new_target_price = -abs(np.random.normal(0.05, 0.02))  # Отрицательное изменение
                                new_target_direction = 'down'
                            else:  # neutral
                                new_target_price = np.random.normal(0, 0.005)  # Близко к нулю
                                new_target_direction = 'neutral'

                            features.append(new_features)
                            target_prices.append(new_target_price)
                            target_directions.append(new_target_direction)
                            valid_records += 1

                logger.info(
                    f"Final target direction distribution: {dict(zip(*np.unique(target_directions, return_counts=True)))}")

                # Используем общий метод обучения
                await self._train_with_data(features, target_prices, target_directions, "periodic_database")

            except Exception as e:
                logger.error(f"Error in model training: {e}", exc_info=True)

            # Ждем до следующего цикла обучения (сокращаем интервал для тестирования)
            await asyncio.sleep(min(interval_hours * 3600, 1800))  # Максимум 30 минут между обучениями
