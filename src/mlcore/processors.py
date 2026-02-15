import json
import talib
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod



class BaseProcessor(ABC):
    def __init__(self):
        self.prev_df = None

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Метод который создаёт фичи из каждого потока, у каждого класса он должен быть свой 
        """
        pass

    def __call__(self, stream_df: pd.DataFrame, need_prev_hour: bool = False):
        # приводим временную метку к читаемому формату
        stream_df['exchange_ts'] = pd.to_datetime(stream_df['exchange_ts'], unit='ms')

        # Сохраняем временную метку для разделения 
        split_index = stream_df['exchange_ts'].iloc[0]

        # Если нужно объединять с предыдущим часом
        if need_prev_hour:
            if self.prev_df is not None:
                df_combined = pd.concat([self.prev_df, stream_df], ignore_index=True)
            else:
                df_combined = stream_df.copy()

            self.prev_df = stream_df.copy()
        else:
            df_combined = stream_df.copy()

        # Устанавливаем временную метку как индекс 
        df_combined.set_index('exchange_ts', inplace=True)

        # Рассчитываем фичи
        df_features = self.create_features(df_combined)

        # Возвращаем временную метку как колонку
        df_features.reset_index(inplace=True)

        # Отрезаем предыдущий час
        result = df_features.loc[df_features['exchange_ts'] >= split_index]

        return result
    


class AggTradesProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        # Сортируем по времени
        df_combined = df_combined.sort_index()
        
        # Получаем 1-минутные свечи
        ohlcv_5s = df_combined['price'].resample('5s', closed='left', label='right').ohlc()
        volume_5s = df_combined['qty'].resample('5s', closed='left', label='right').sum()
        ohlcv_5s['volume'] = volume_5s

        # Создаём датафрейм для добавления технических индикаторов и осцилляторов
        df = ohlcv_5s.copy()

        return df
    


class RawTradesProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        # Сортируем по времени
        df_combined = df_combined.sort_index()
        # Отавляем только нужные колонки
        df = df_combined[['isBuyerMaker', 'qty', 'price']].copy()

        df['buy_qty'] = np.where(~df['isBuyerMaker'], df['qty'], 0.0)
        df['sell_qty'] = np.where(df['isBuyerMaker'], df['qty'], 0.0)
        df['is_aggressive_buy'] = (~df['isBuyerMaker']).astype(int)  # агрессивные покупки (берут ask)
        df['is_aggressive_sell'] = df['isBuyerMaker'].astype(int)     # агрессивные продажи (берут bid)

        qty = df['qty']
        price = df['price']
        buy_qty = df['buy_qty']
        sell_qty = df['sell_qty']
        aggressive_buy = df['is_aggressive_buy']
        aggressive_sell = df['is_aggressive_sell']
        price_vol = price * qty

        windows_sec = [1,2,3,5,10,15,20,30,45,60,90,120]
        feature_dict = {}

        for win_sec in windows_sec:
            win = f"{win_sec}s"

            # Rolling windows (только прошлое)
            r_qty = qty.rolling(win, closed='left')
            r_buy = buy_qty.rolling(win, closed='left')
            r_sell = sell_qty.rolling(win, closed='left')
            r_price = price.rolling(win, closed='left')
            r_vol = price_vol.rolling(win, closed='left')
            r_agg_buy = aggressive_buy.rolling(win, closed='left')
            r_agg_sell = aggressive_sell.rolling(win, closed='left')

            # === 1. Базовые объемы ===
            total_vol = r_qty.sum()
            buy_vol = r_buy.sum()
            sell_vol = r_sell.sum()

            # === 2. Дисбалансы ===
            vol_imbalance = (buy_vol - sell_vol) / (total_vol + 1e-12)

            # === 3. Интенсивность сделок ===
            n_trades = r_qty.count()
            trade_freq = n_trades / win_sec

            # === 4. VWAP и отклонения ===
            vwap = r_vol.sum() / (total_vol + 1e-12)
            vwap_dev = price - vwap
            vwap_dev_norm = vwap_dev / (r_price.std() + 1e-12)

            # === 6. Агрессивность ===
            agg_buy_count = r_agg_buy.sum()
            agg_sell_count = r_agg_sell.sum()
            agg_imbalance = (agg_buy_count - agg_sell_count) / (n_trades + 1e-12)


            # === Сохранение всех фич ===
            feature_dict[f'volume_{win}'] = total_vol
            feature_dict[f'buy_volume_{win}'] = buy_vol
            feature_dict[f'sell_volume_{win}'] = sell_vol
            feature_dict[f'volume_imbalance_{win}'] = vol_imbalance

            feature_dict[f'n_trades_{win}'] = n_trades
            feature_dict[f'trade_freq_{win}'] = trade_freq

            feature_dict[f'vwap_{win}'] = vwap
            feature_dict[f'vwap_dev_{win}'] = vwap_dev
            feature_dict[f'vwap_dev_norm_{win}'] = vwap_dev_norm

            feature_dict[f'agg_buy_count_{win}'] = agg_buy_count
            feature_dict[f'agg_sell_count_{win}'] = agg_sell_count
            feature_dict[f'agg_imbalance_{win}'] = agg_imbalance

        features_df = pd.DataFrame(feature_dict, index=df.index)
        
        # Ресемплируем к 1-секундной сетке
        features_df = (
            features_df
            .resample('5s', closed='left', label='right')
            .last()
            .ffill()
        )

        return features_df
    


class OrderbookProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        """
        df_combined: long-format DataFrame с колонками:
            ['exchange_ts', 'side', 'level', 'price', 'qty']
        """
        # Отавляем только нужные колонки
        df = df_combined[['side', 'level', 'price', 'qty']].copy()

        # ================ Преобразуем данные из long формата в numpy матрицы ================
        # Разделяем bids и asks
        bids = df[df['side'] == 'bid'][['price', 'qty', 'level']]
        asks = df[df['side'] == 'ask'][['price', 'qty', 'level']]

        # Пивотим по exchange_ts (индекс уже установлен)
        bid_piv = bids.pivot_table(
            index=bids.index,
            columns='level',
            values=['price', 'qty'],
        ).resample('5s', closed='left', label='right').last().ffill()

        ask_piv = asks.pivot_table(
            index=asks.index,
            columns='level',
            values=['price', 'qty'],
        ).resample('5s', closed='left', label='right').last().ffill()

        # Убираем мультииндекс у колонок
        bid_piv.columns = [f"bid_{col[0]}_{col[1]}" for col in bid_piv.columns]
        ask_piv.columns = [f"ask_{col[0]}_{col[1]}" for col in ask_piv.columns]

        wide_df = pd.concat([bid_piv, ask_piv], axis=1)

        return wide_df