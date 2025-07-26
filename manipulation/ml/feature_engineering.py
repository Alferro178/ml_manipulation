import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, trade_processor=None, orderbook_analyzer=None):
        self.trade_processor = trade_processor
        self.orderbook_analyzer = orderbook_analyzer
        logger.info("FeatureEngineer initialized.")

    def extract_features(self, symbol: str, order_book_snapshot: Dict, trade_history: List[Dict],
                         timestamp_sec: float) -> Optional[Dict[str, Any]]:
        """
        Extracts features from order book and trade data for a given symbol at a specific timestamp.
        Args:
            symbol: The trading pair symbol (e.g., "BTCUSDT").
            order_book_snapshot: A dictionary with 'bids' and 'asks' lists.
            trade_history: A list of trade dictionaries.
            timestamp_sec: The timestamp (in seconds) around which to extract features.
        Returns:
            A dictionary of extracted features, or None if data is insufficient.
        """
        logger.debug(
            f"Extracting features for {symbol}: snapshot={bool(order_book_snapshot)}, trades={len(trade_history)}")

        features = {}

        # --- Order Book Features ---
        bids = order_book_snapshot.get('bids', []) if order_book_snapshot else []
        asks = order_book_snapshot.get('asks', []) if order_book_snapshot else []

        if not bids or not asks:
            logger.warning(f"Order book snapshot is empty or invalid for {symbol}.")
            return None  # Возвращаем None если нет данных orderbook
        else:
            try:
                best_bid_price = bids[0][0]
                best_bid_size = bids[0][1]
                best_ask_price = asks[0][0]
                best_ask_size = asks[0][1]
            except (IndexError, TypeError):
                logger.warning(f"Invalid orderbook structure for {symbol}")
                return None

            features['spread'] = best_ask_price - best_bid_price
            features['mid_price'] = (best_bid_price + best_ask_price) / 2
            features['relative_spread'] = features['spread'] / features['mid_price'] if features['mid_price'] > 0 else 0

            # Liquidity at different depths
            depth_levels = [5, 10, 20]
            for level in depth_levels:
                try:
                    features[f'bid_liquidity_depth_{level}'] = sum(s for p, s in bids[:level])
                    features[f'ask_liquidity_depth_{level}'] = sum(s for p, s in asks[:level])
                except (TypeError, ValueError):
                    features[f'bid_liquidity_depth_{level}'] = 0.0
                    features[f'ask_liquidity_depth_{level}'] = 0.0

                features[f'total_liquidity_depth_{level}'] = features[f'bid_liquidity_depth_{level}'] + features[
                    f'ask_liquidity_depth_{level}']
                features[f'liquidity_imbalance_depth_{level}'] = (features[f'bid_liquidity_depth_{level}'] - features[
                    f'ask_liquidity_depth_{level}']) / features[f'total_liquidity_depth_{level}'] if features[
                                                                                                         f'total_liquidity_depth_{level}'] > 0 else 0

            # Order book imbalance (overall)
            try:
                total_bid_volume = sum(s for p, s in bids)
                total_ask_volume = sum(s for p, s in asks)
            except (TypeError, ValueError):
                total_bid_volume = total_ask_volume = 0

            total_ob_volume = total_bid_volume + total_ask_volume
            features['orderbook_imbalance'] = (
                                                          total_bid_volume - total_ask_volume) / total_ob_volume if total_ob_volume > 0 else 0

        # --- Trade Features (recent history) ---
        # Увеличиваем окно для поиска недавних сделок
        recent_trades = [trade for trade in trade_history if
                         timestamp_sec - trade.get('timestamp', 0) <= 300] if trade_history else []

        if not recent_trades:
            # Если нет недавних сделок, используем исторические средние значения
            logger.debug(f"No recent trades for feature engineering for {symbol}, using historical averages.")

            # Используем все доступные сделки для расчета средних значений
            all_trades = trade_history if trade_history else []
            if all_trades:
                total_volume = sum(trade.get('volume_usdt', 0) for trade in all_trades)
                buy_volume = sum(trade.get('volume_usdt', 0) for trade in all_trades if trade.get('side') == 'Buy')
                sell_volume = sum(trade.get('volume_usdt', 0) for trade in all_trades if trade.get('side') == 'Sell')

                features['total_trade_volume_usd'] = total_volume / len(all_trades)  # Средний объем
                features['buy_trade_volume_usd'] = buy_volume / len(all_trades)
                features['sell_trade_volume_usd'] = sell_volume / len(all_trades)
                features['trade_count'] = len(all_trades)
                features['trade_volume_imbalance'] = (
                                                                 buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
                features['avg_trade_size_usd'] = total_volume / len(all_trades) if len(all_trades) > 0 else 0

                prices = [trade.get('price', 0) for trade in all_trades if trade.get('price', 0) > 0]
                if len(prices) > 1:
                    features['price_volatility'] = np.std(prices)
                    features['price_change_recent'] = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                else:
                    features['price_volatility'] = 0.0
                    features['price_change_recent'] = 0.0
            else:
                # Полностью нулевые значения только если нет вообще никаких данных
                features['total_trade_volume_usd'] = 0.0
                features['buy_trade_volume_usd'] = 0.0
                features['sell_trade_volume_usd'] = 0.0
                features['trade_count'] = 0
                features['trade_volume_imbalance'] = 0.0
                features['avg_trade_size_usd'] = 0.0
                features['price_volatility'] = 0.0
                features['price_change_recent'] = 0.0
        else:
            total_trade_volume_usd = sum(trade.get('volume_usdt', 0) for trade in recent_trades)
            buy_volume_usd = sum(trade.get('volume_usdt', 0) for trade in recent_trades if trade.get('side') == 'Buy')
            sell_volume_usd = sum(trade.get('volume_usdt', 0) for trade in recent_trades if trade.get('side') == 'Sell')

            features['total_trade_volume_usd'] = total_trade_volume_usd
            features['buy_trade_volume_usd'] = buy_volume_usd
            features['sell_trade_volume_usd'] = sell_volume_usd
            features['trade_count'] = len(recent_trades)

            features['trade_volume_imbalance'] = (
                                                             buy_volume_usd - sell_volume_usd) / total_trade_volume_usd if total_trade_volume_usd > 0 else 0
            features['avg_trade_size_usd'] = total_trade_volume_usd / len(recent_trades) if len(
                recent_trades) > 0 else 0

            recent_prices = [trade.get('price', 0) for trade in recent_trades if trade.get('price', 0) > 0]
            if len(recent_prices) > 1:
                features['price_volatility'] = np.std(recent_prices)
                features['price_change_recent'] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if \
                recent_prices[0] > 0 else 0
            else:
                features['price_volatility'] = 0.0
                features['price_change_recent'] = 0.0

        # --- Interaction Features ---
        if self.trade_processor and symbol in self.trade_processor.processed_trades:
            features['wash_trade_signal'] = int(self.trade_processor.wash_trade_alerts.get(symbol, 0) > 0)
            features['ping_pong_signal'] = int(self.trade_processor.ping_pong_alerts.get(symbol, 0) > 0)
            features['ramping_signal'] = int(self.trade_processor.ramping_alerts.get(symbol, 0) > 0)
        else:
            features['wash_trade_signal'] = 0
            features['ping_pong_signal'] = 0
            features['ramping_signal'] = 0

        if self.orderbook_analyzer and symbol in self.orderbook_analyzer.analysis_counts:
            features['iceberg_signal'] = int(self.orderbook_analyzer.alert_counts.get(symbol, 0) > 0)
            features['layering_spoofing_signal'] = int(self.orderbook_analyzer.alert_counts.get(symbol, 0) > 0)
        else:
            features['iceberg_signal'] = 0
            features['layering_spoofing_signal'] = 0

        logger.debug(f"Extracted features for {symbol}: {features}")
        return features

    def get_feature_names(self) -> List[str]:
        """Возвращает список всех возможных имен признаков."""
        dummy_symbol = "BTCUSDT"
        dummy_order_book_snapshot = {
            'bids': [[99.9, 100], [99.8, 200]],
            'asks': [[100.1, 150], [100.2, 250]]
        }
        dummy_trade_history = [
            {'timestamp': 1, 'price': 100, 'size': 10, 'side': 'Buy', 'volume_usdt': 1000},
            {'timestamp': 2, 'price': 100.01, 'size': 12, 'side': 'Sell', 'volume_usdt': 1200}
        ]
        dummy_timestamp_sec = time.time()

        dummy_features = self.extract_features(dummy_symbol, dummy_order_book_snapshot, dummy_trade_history,
                                               dummy_timestamp_sec)
        if dummy_features:
            return sorted(list(dummy_features.keys()))
        return []