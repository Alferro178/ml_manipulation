
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class MLModel:
    """
    Улучшенная модель машинного обучения с поддержкой:
    - Онлайн обучения на потоковых данных
    - Нормализации данных разного масштаба
    - Инкрементального обучения
    - Автоматического сохранения и загрузки
    - Адаптивного обучения
    """
    
    def __init__(self, model_path="improved_ml_model.joblib", 
                 incremental_learning=True,
                 auto_retrain_threshold=1000,
                 model_type="hybrid"):
        self.model_path = model_path
        self.incremental_learning = incremental_learning
        self.auto_retrain_threshold = auto_retrain_threshold
        self.model_type = model_type  # "hybrid", "random_forest", "neural_network"
        
        # Модели для разных задач
        self.price_model: Optional[Any] = None
        self.direction_model: Optional[Any] = None
        
        # Онлайн модели для инкрементального обучения
        self.online_price_model: Optional[SGDRegressor] = None
        self.online_direction_model: Optional[SGDClassifier] = None
        
        # Скейлеры для нормализации данных разного масштаба
        self.robust_scaler: Optional[RobustScaler] = None
        self.standard_scaler: Optional[StandardScaler] = None
        
        # Метаданные модели
        self.feature_names: Optional[List[str]] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.training_stats: Dict[str, Any] = {
            'samples_processed': 0,
            'last_training_time': None,
            'model_version': 1,
            'accuracy_history': deque(maxlen=100),
            'error_history': deque(maxlen=100)
        }
        
        # Буфер для накопления данных
        self.training_buffer = deque(maxlen=10000)
        self.prediction_cache = {}
        
        # Блокировки для потокобезопасности
        self.model_lock = threading.RLock()
        self.buffer_lock = threading.Lock()
        
        # Инициализация
        self._load_models()
        self._initialize_online_models()
        logger.info("MLModel initialized successfully.")

    # ... keep existing code (all methods from _initialize_online_models to reset_models remain the same)

    def _initialize_online_models(self):
        """Инициализация моделей для онлайн обучения"""
        try:
            # SGD модели для инкрементального обучения
            self.online_price_model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                alpha=0.001,
                random_state=42,
                warm_start=True
            )
            
            self.online_direction_model = SGDClassifier(
                learning_rate='adaptive',
                eta0=0.01,
                alpha=0.001,
                random_state=42,
                warm_start=True,
                loss='log_loss'  # Обновлено с 'log' на 'log_loss'
            )
            
            # Робастный скейлер для данных разного масштаба
            self.robust_scaler = RobustScaler()
            self.standard_scaler = StandardScaler()
            
            logger.info("Online models initialized.")
            
        except Exception as e:
            logger.error(f"Failed to initialize online models: {e}")

    def _load_models(self):
        """Загрузка предобученных моделей"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    models_data = pickle.load(f)
                
                self.price_model = models_data.get('price_model')
                self.direction_model = models_data.get('direction_model')
                self.online_price_model = models_data.get('online_price_model')
                self.online_direction_model = models_data.get('online_direction_model')
                self.robust_scaler = models_data.get('robust_scaler')
                self.standard_scaler = models_data.get('standard_scaler')
                self.feature_names = models_data.get('feature_names')
                self.feature_importance = models_data.get('feature_importance')
                self.training_stats = models_data.get('training_stats', self.training_stats)
                
                logger.info(f"Models loaded successfully. Version: {self.training_stats.get('model_version', 1)}")
                
            except Exception as e:
                logger.warning(f"Failed to load models: {e}. Initializing new models.")
                self._initialize_new_models()
        else:
            logger.info("No existing models found. Initializing new models.")
            self._initialize_new_models()

    def _initialize_new_models(self):
        """Инициализация новых моделей"""
        try:
            if self.model_type == "random_forest":
                self.price_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    warm_start=True
                )
                
                self.direction_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    warm_start=True
                )
                
            elif self.model_type == "neural_network":
                self.price_model = MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    warm_start=True,
                    early_stopping=True
                )
                
                self.direction_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50, 25),
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    warm_start=True,
                    early_stopping=True
                )
                
            else:  # hybrid
                # Используем Random Forest как основную модель
                self.price_model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.direction_model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            logger.info(f"New {self.model_type} models initialized.")
            
        except Exception as e:
            logger.error(f"Failed to initialize new models: {e}")

    def add_training_data(self, features: Dict[str, float], 
                         target_price: float, 
                         target_direction: int):
        """Добавление данных в буфер для обучения"""
        try:
            with self.buffer_lock:
                # Нормализация данных для разных масштабов
                normalized_features = self._normalize_features(features)
                
                training_sample = {
                    'features': normalized_features,
                    'target_price': target_price,
                    'target_direction': target_direction,
                    'timestamp': time.time()
                }
                
                self.training_buffer.append(training_sample)
                self.training_stats['samples_processed'] += 1
                
                # Автоматическое обучение при достижении порога
                if (len(self.training_buffer) >= self.auto_retrain_threshold and 
                    self.incremental_learning):
                    self._incremental_train()
                    
        except Exception as e:
            logger.error(f"Error adding training data: {e}")

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Нормализация признаков для работы с данными разного масштаба"""
        try:
            # Применяем логарифмическое масштабирование для цен
            normalized_features = {}
            
            for key, value in features.items():
                if 'price' in key.lower() or 'volume' in key.lower():
                    # Для цен и объемов используем логарифмическое масштабирование
                    if value > 0:
                        normalized_features[key] = np.log1p(value)
                    else:
                        normalized_features[key] = 0
                elif 'spread' in key.lower() or 'ratio' in key.lower():
                    # Для спредов и отношений используем стандартную нормализацию
                    normalized_features[key] = value
                else:
                    # Для остальных признаков
                    normalized_features[key] = value
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features

    def _incremental_train(self):
        """Инкрементальное обучение на накопленных данных"""
        try:
            with self.model_lock:
                if len(self.training_buffer) < 10:
                    return
                
                # Подготовка данных
                features_list = []
                price_targets = []
                direction_targets = []
                
                for sample in list(self.training_buffer):
                    features_list.append(list(sample['features'].values()))
                    price_targets.append(sample['target_price'])
                    direction_targets.append(sample['target_direction'])
                
                X = np.array(features_list)
                y_price = np.array(price_targets)
                y_direction = np.array(direction_targets)
                
                # Обновление скейлеров
                if self.robust_scaler is not None:
                    X_scaled = self.robust_scaler.partial_fit(X).transform(X)
                else:
                    self.robust_scaler = RobustScaler()
                    X_scaled = self.robust_scaler.fit_transform(X)
                
                # Инкрементальное обучение онлайн моделей
                if self.online_price_model is not None and self.online_direction_model is not None:
                    # Обучение модели предсказания цены
                    self.online_price_model.partial_fit(X_scaled, y_price)
                    
                    # Обучение модели направления
                    unique_classes = np.unique(y_direction)
                    if len(unique_classes) > 1:
                        self.online_direction_model.partial_fit(X_scaled, y_direction, 
                                                              classes=np.array([-1, 0, 1]))
                
                # Обновление статистики
                self.training_stats['last_training_time'] = datetime.now().isoformat()
                
                # Очистка буфера
                self.training_buffer.clear()
                
                logger.info(f"Incremental training completed. Samples processed: {len(features_list)}")
                
        except Exception as e:
            logger.error(f"Error in incremental training: {e}")

    def train(self, X: np.ndarray, y_price: np.ndarray, y_direction: np.ndarray) -> Dict[str, float]:
        """Полное обучение моделей"""
        try:
            with self.model_lock:
                if len(X) < 10:
                    logger.warning("Insufficient data for training")
                    return {}
                
                # Разделение данных
                X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
                    X, y_price, y_direction, test_size=0.2, random_state=42
                )
                
                # Масштабирование данных
                self.robust_scaler = RobustScaler()
                X_train_scaled = self.robust_scaler.fit_transform(X_train)
                X_test_scaled = self.robust_scaler.transform(X_test)
                
                # Обучение основных моделей
                if self.price_model is not None:
                    self.price_model.fit(X_train_scaled, y_price_train)
                    price_pred = self.price_model.predict(X_test_scaled)
                    price_mse = mean_squared_error(y_price_test, price_pred)
                    price_mae = mean_absolute_error(y_price_test, price_pred)
                else:
                    price_mse = price_mae = float('inf')
                
                if self.direction_model is not None:
                    self.direction_model.fit(X_train_scaled, y_dir_train)
                    direction_pred = self.direction_model.predict(X_test_scaled)
                    direction_acc = accuracy_score(y_dir_test, direction_pred)
                else:
                    direction_acc = 0.0
                
                # Инициализация онлайн моделей
                if self.online_price_model is not None:
                    self.online_price_model.fit(X_train_scaled, y_price_train)
                
                if self.online_direction_model is not None:
                    self.online_direction_model.fit(X_train_scaled, y_dir_train)
                
                # Обновление статистики
                metrics = {
                    'price_mse': price_mse,
                    'price_mae': price_mae,
                    'direction_accuracy': direction_acc
                }
                
                self.training_stats['accuracy_history'].append(direction_acc)
                self.training_stats['error_history'].append(price_mse)
                self.training_stats['last_training_time'] = datetime.now().isoformat()
                self.training_stats['model_version'] += 1
                
                # Сохранение модели
                self.save_models()
                
                logger.info(f"Training completed. Metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error in training: {e}")
            return {}

    def predict(self, features: Dict[str, float]) -> Tuple[float, str, float]:
        """Предсказание цены и направления с доверительной оценкой"""
        try:
            with self.model_lock:
                # Нормализация входных данных
                normalized_features = self._normalize_features(features)
                feature_vector = np.array([list(normalized_features.values())])
                
                # Применение скейлера
                if self.robust_scaler is not None:
                    feature_vector = self.robust_scaler.transform(feature_vector)
                
                # Предсказание цены
                price_change = 0.0
                if self.online_price_model is not None:
                    try:
                        price_change = self.online_price_model.predict(feature_vector)[0]
                    except:
                        if self.price_model is not None:
                            price_change = self.price_model.predict(feature_vector)[0]
                
                # Предсказание направления
                direction = "neutral"
                confidence = 0.0
                
                if self.online_direction_model is not None:
                    try:
                        direction_pred = self.online_direction_model.predict(feature_vector)[0]
                        direction_proba = self.online_direction_model.predict_proba(feature_vector)[0]
                        confidence = np.max(direction_proba)
                        
                        if direction_pred == 1:
                            direction = "up"
                        elif direction_pred == -1:
                            direction = "down"
                        else:
                            direction = "neutral"
                    except:
                        if self.direction_model is not None:
                            direction_pred = self.direction_model.predict(feature_vector)[0]
                            if hasattr(self.direction_model, 'predict_proba'):
                                direction_proba = self.direction_model.predict_proba(feature_vector)[0]
                                confidence = np.max(direction_proba)
                            
                            if direction_pred == 1:
                                direction = "up"
                            elif direction_pred == -1:
                                direction = "down"
                
                return price_change, direction, confidence
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, "neutral", 0.0

    def save_models(self):
        """Сохранение всех моделей и метаданных"""
        try:
            with self.model_lock:
                models_data = {
                    'price_model': self.price_model,
                    'direction_model': self.direction_model,
                    'online_price_model': self.online_price_model,
                    'online_direction_model': self.online_direction_model,
                    'robust_scaler': self.robust_scaler,
                    'standard_scaler': self.standard_scaler,
                    'feature_names': self.feature_names,
                    'feature_importance': self.feature_importance,
                    'training_stats': self.training_stats
                }
                
                # Создаем резервную копию
                if os.path.exists(self.model_path):
                    backup_path = f"{self.model_path}.backup"
                    os.rename(self.model_path, backup_path)
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(models_data, f)
                
                logger.info(f"Models saved to {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            'model_type': self.model_type,
            'training_stats': self.training_stats,
            'feature_names': self.feature_names,
            'buffer_size': len(self.training_buffer),
            'model_path': self.model_path
        }

    def reset_models(self):
        """Сброс и переинициализация моделей"""
        with self.model_lock:
            self.training_buffer.clear()
            self.prediction_cache.clear()
            self.training_stats = {
                'samples_processed': 0,
                'last_training_time': None,
                'model_version': 1,
                'accuracy_history': deque(maxlen=100),
                'error_history': deque(maxlen=100)
            }
            self._initialize_new_models()
            self._initialize_online_models()
            logger.info("Models reset and reinitialized")
