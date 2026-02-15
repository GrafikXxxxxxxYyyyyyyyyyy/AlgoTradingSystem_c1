import json
import talib
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Optional


class BaseProcessor(ABC):
    """
    grid_resolution_ms: интервал сетки в миллисекундах (5000 = 5 s, 10000 = 10 s, 100 = 100 ms). None = тиковый режим.
    """
    def __init__(self, grid_resolution_ms: Optional[int] = 5000):
        self.prev_df = None
        self._grid_value = grid_resolution_ms
        if grid_resolution_ms is not None:
            self._rule_ns = int(grid_resolution_ms * 1_000_000)
            self._resample_rule = f"{int(grid_resolution_ms)}ms"
        else:
            self._rule_ns = None
            self._resample_rule = None

    def _round_index_ns_to_grid(self, right_edge_ns: np.ndarray) -> np.ndarray:
        """Округляет границы бинов до шага сетки (rule_ns), чтобы индексы совпадали при мерже (одна строка на интервал)."""
        if self._rule_ns is None:
            return right_edge_ns
        return (right_edge_ns // self._rule_ns) * self._rule_ns

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Метод который создаёт фичи из каждого потока, у каждого класса он должен быть свой 
        """
        pass

    def __call__(
        self,
        stream_df: pd.DataFrame,
        need_prev_hour: bool = False,
        keep_ts_range: Optional[tuple] = None,
        **kwargs,
    ):
        """
        keep_ts_range: (t_min, t_max) — оставить только строки с exchange_ts в этом диапазоне (включительно).
        Используется при обработке по файлам с перекрытием: в stream_df уже лежит tail_prev + current + head_next.
        """
        stream_df = stream_df.copy()
        stream_df["exchange_ts"] = pd.to_datetime(stream_df["exchange_ts"], unit="ms")

        split_index = stream_df["exchange_ts"].iloc[0]

        if need_prev_hour:
            if self.prev_df is not None:
                df_combined = pd.concat([self.prev_df, stream_df], ignore_index=True)
            else:
                df_combined = stream_df
            self.prev_df = stream_df.copy()
        else:
            df_combined = stream_df

        df_combined.set_index("exchange_ts", inplace=True)
        df_features = self.create_features(df_combined)
        df_features.reset_index(inplace=True)

        if keep_ts_range is not None:
            t_lo, t_hi = keep_ts_range
            mask = (df_features["exchange_ts"] >= t_lo) & (df_features["exchange_ts"] <= t_hi)
            return df_features.loc[mask]
        if need_prev_hour:
            return df_features.loc[df_features["exchange_ts"] >= split_index]
        return df_features
    


class AggTradesProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        df_combined = df_combined.sort_index()

        if self._rule_ns is not None:
            # OHLCV через группировку по bin (numpy + reduceat) вместо pandas resample
            t_ns = df_combined.index.astype(np.int64)
            price = np.asarray(df_combined["price"], dtype=np.float64)
            qty = np.asarray(df_combined["qty"], dtype=np.float64)

            t_min = t_ns.min()
            rule_ns = self._rule_ns
            bin_id = (t_ns - t_min) // rule_ns

            order = np.argsort(bin_id)
            bin_sorted = bin_id[order]
            price_sorted = price[order]
            qty_sorted = qty[order]

            unique_bins, start_idx = np.unique(bin_sorted, return_index=True)
            end_idx = np.concatenate([start_idx[1:], [len(bin_sorted)]])

            open_ = price_sorted[start_idx]
            close = price_sorted[end_idx - 1]
            high = np.maximum.reduceat(price_sorted, start_idx)
            low = np.minimum.reduceat(price_sorted, start_idx)
            volume = np.add.reduceat(qty_sorted, start_idx)

            right_edge_ns = t_min + (unique_bins + 1) * rule_ns
            right_edge_ns = self._round_index_ns_to_grid(right_edge_ns)
            index = pd.to_datetime(right_edge_ns, unit="ns")
            index.name = "exchange_ts"

            df = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            )
        else:
            df = df_combined[["price", "qty"]].copy()
            df["open"] = df["high"] = df["low"] = df["close"] = df["price"]
            df["volume"] = df["qty"]
            df = df[["open", "high", "low", "close", "volume"]]

        return df
    


class RawTradesProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        """
        Векторизованный расчёт rolling-фичей по времени (cumsum + searchsorted)
        вместо 12 отдельных pandas rolling — сохраняет логику и результат.
        """
        df_combined = df_combined.sort_index()
        df = df_combined[["isBuyerMaker", "qty", "price"]].copy()

        is_buy = ~df["isBuyerMaker"].values
        qty = np.asarray(df["qty"], dtype=np.float64)
        price = np.asarray(df["price"], dtype=np.float64)

        buy_qty = np.where(is_buy, qty, 0.0)
        sell_qty = np.where(is_buy, 0.0, qty)
        aggressive_buy = np.where(is_buy, 1, 0).astype(np.float64)
        aggressive_sell = np.where(is_buy, 0, 1).astype(np.float64)
        price_vol = price * qty

        # Время в секундах для окон
        t_sec = df.index.astype(np.int64) / 1e9
        n = len(t_sec)

        # Кумулятивные суммы (с нулём в начале для суммы на [left_idx, i))
        def _cs(x):
            out = np.empty(n + 1, dtype=np.float64)
            out[0] = 0
            np.cumsum(x, out=out[1:])
            return out

        cs_qty = _cs(qty)
        cs_buy = _cs(buy_qty)
        cs_sell = _cs(sell_qty)
        cs_pv = _cs(price_vol)
        cs_agg_buy = _cs(aggressive_buy)
        cs_agg_sell = _cs(aggressive_sell)
        cs_price = _cs(price)
        cs_price2 = _cs(price * price)

        windows_sec = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
        feature_dict = {}

        for win_sec in windows_sec:
            win = f"{win_sec}s"
            # Левая граница окна [t - win_sec, t) для каждой строки (closed='left')
            left_idx = np.searchsorted(t_sec, t_sec - win_sec, side="left")
            left_idx = np.maximum(left_idx, 0)

            # Сумма на [left_idx[i], i) = cumsum[i] - cumsum[left_idx[i]]
            total_vol = cs_qty[:n] - cs_qty[left_idx]
            buy_vol = cs_buy[:n] - cs_buy[left_idx]
            sell_vol = cs_sell[:n] - cs_sell[left_idx]
            n_trades = np.arange(n, dtype=np.float64) - left_idx.astype(np.float64)
            n_trades = np.maximum(n_trades, 0)

            vol_imbalance = (buy_vol - sell_vol) / (total_vol + 1e-12)

            trade_freq = n_trades / (win_sec + 1e-12)

            vwap_num = cs_pv[:n] - cs_pv[left_idx]
            vwap = vwap_num / (total_vol + 1e-12)
            vwap_dev = price - vwap

            sum_price = cs_price[:n] - cs_price[left_idx]
            sum_price2 = cs_price2[:n] - cs_price2[left_idx]
            mean_p = sum_price / (n_trades + 1e-12)
            var_p = sum_price2 / (n_trades + 1e-12) - mean_p * mean_p
            std_p = np.sqrt(np.maximum(var_p, 0.0))
            vwap_dev_norm = vwap_dev / (std_p + 1e-12)

            agg_buy_count = cs_agg_buy[:n] - cs_agg_buy[left_idx]
            agg_sell_count = cs_agg_sell[:n] - cs_agg_sell[left_idx]
            agg_imbalance = (agg_buy_count - agg_sell_count) / (n_trades + 1e-12)

            feature_dict[f"volume_{win}"] = total_vol
            feature_dict[f"buy_volume_{win}"] = buy_vol
            feature_dict[f"sell_volume_{win}"] = sell_vol
            feature_dict[f"volume_imbalance_{win}"] = vol_imbalance
            feature_dict[f"n_trades_{win}"] = n_trades
            feature_dict[f"trade_freq_{win}"] = trade_freq
            feature_dict[f"vwap_{win}"] = vwap
            feature_dict[f"vwap_dev_{win}"] = vwap_dev
            feature_dict[f"vwap_dev_norm_{win}"] = vwap_dev_norm
            feature_dict[f"agg_buy_count_{win}"] = agg_buy_count
            feature_dict[f"agg_sell_count_{win}"] = agg_sell_count
            feature_dict[f"agg_imbalance_{win}"] = agg_imbalance

        features_df = pd.DataFrame(feature_dict, index=df.index)

        if self._resample_rule is not None:
            features_df = (
                features_df
                .resample(self._resample_rule, closed="left", label="right")
                .last()
                .ffill()
            )
            # Выравниваем индекс по шагу сетки (как в AggTrades/Orderbook), иначе при мерже появятся лишние строки
            index_ns = features_df.index.astype(np.int64)
            features_df.index = pd.to_datetime(
                self._round_index_ns_to_grid(index_ns), unit="ns"
            )
            # После округления возможны дубли индекса — оставляем последнюю строку на каждый момент
            features_df = features_df[~features_df.index.duplicated(keep="last")]

        return features_df
    


class OrderbookProcessor(BaseProcessor):
    def create_features(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        """
        df_combined: long-format DataFrame с колонками:
            ['exchange_ts', 'side', 'level', 'price', 'qty']
        Long -> wide через numpy; маппинг ts->row через np.unique; ресэмпл — через бинирование.
        """
        df = df_combined[["side", "level", "price", "qty"]]
        idx_values = df.index.values
        uniq_ts, row_idx = np.unique(idx_values, return_inverse=True)
        n_ts = len(uniq_ts)
        n_levels = int(df["level"].max()) + 1

        bid_price = np.full((n_ts, n_levels), np.nan, dtype=np.float64)
        bid_qty = np.full((n_ts, n_levels), np.nan, dtype=np.float64)
        ask_price = np.full((n_ts, n_levels), np.nan, dtype=np.float64)
        ask_qty = np.full((n_ts, n_levels), np.nan, dtype=np.float64)

        lvl = df["level"].values.astype(np.intp)
        prc = df["price"].values.astype(np.float64)
        qty = df["qty"].values.astype(np.float64)
        side_vals = df["side"].values
        bid_mask = side_vals == "bid"
        ask_mask = side_vals == "ask"

        flat_bid = row_idx[bid_mask] * n_levels + lvl[bid_mask]
        flat_ask = row_idx[ask_mask] * n_levels + lvl[ask_mask]
        np.put(bid_price, flat_bid, prc[bid_mask])
        np.put(bid_qty, flat_bid, qty[bid_mask])
        np.put(ask_price, flat_ask, prc[ask_mask])
        np.put(ask_qty, flat_ask, qty[ask_mask])

        n_cols = 4 * n_levels
        wide = np.empty((n_ts, n_cols), dtype=np.float64)
        wide[:, :n_levels] = bid_price
        wide[:, n_levels : 2 * n_levels] = bid_qty
        wide[:, 2 * n_levels : 3 * n_levels] = ask_price
        wide[:, 3 * n_levels :] = ask_qty

        bid_cols = [f"bid_price_{i}" for i in range(n_levels)] + [f"bid_qty_{i}" for i in range(n_levels)]
        ask_cols = [f"ask_price_{i}" for i in range(n_levels)] + [f"ask_qty_{i}" for i in range(n_levels)]
        index = pd.to_datetime(uniq_ts)
        index.name = "exchange_ts"

        if self._rule_ns is not None:
            t_ns = uniq_ts.astype(np.int64)
            t_min = t_ns.min()
            rule_ns = self._rule_ns
            bin_id = (t_ns - t_min) // rule_ns
            order = np.argsort(bin_id)
            bin_sorted = bin_id[order]
            unique_bins, start_idx = np.unique(bin_sorted, return_index=True)
            end_idx = np.concatenate([start_idx[1:], [n_ts]])
            last_row_idx = order[end_idx - 1]
            binned = wide[last_row_idx]
            binned_ffill = pd.DataFrame(binned).ffill().values
            right_edge_ns = t_min + (unique_bins + 1) * rule_ns
            right_edge_ns = self._round_index_ns_to_grid(right_edge_ns)
            index = pd.to_datetime(right_edge_ns, unit="ns")
            index.name = "exchange_ts"
            wide = binned_ffill

        wide_df = pd.DataFrame(wide, index=index, columns=bid_cols + ask_cols)
        return wide_df