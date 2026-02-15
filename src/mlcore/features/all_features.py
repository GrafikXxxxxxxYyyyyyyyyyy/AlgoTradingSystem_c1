import talib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf, erfinv, gammaln, betainc, beta, gamma, digamma



def orderbook_depth_imbalance_velocity(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """Скорость изменения дисбаланса глубины стакана"""
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(n_levels)):
        bid_depth = sum(df[f'bid_qty_{i}'] for i in range(n_levels))
        ask_depth = sum(df[f'ask_qty_{i}'] for i in range(n_levels))
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
        velocity = imbalance.diff().fillna(0)
        return velocity.rolling(window=3, min_periods=1).mean()
    return pd.Series(0.0, index=df.index)

def orderbook_depth_imbalance_change(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """Изменение дисбаланса глубины стакана"""
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(n_levels)):
        bid_depth = sum(df[f'bid_qty_{i}'] for i in range(n_levels))
        ask_depth = sum(df[f'ask_qty_{i}'] for i in range(n_levels))
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
        change = imbalance.diff().fillna(0)
        return change.rolling(window=2, min_periods=1).mean()
    return pd.Series(0.0, index=df.index)

def microprice_divergence_velocity(df: pd.DataFrame, window: int = 2) -> pd.Series:
    """Скорость дивергенции микропрайса от цены закрытия"""
    if 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns or 'bid_qty_0' not in df.columns or 'ask_qty_0' not in df.columns or 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    micro = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-12)
    divergence = micro - df['close']
    velocity = divergence.diff().fillna(0)
    return velocity.rolling(window=window, min_periods=1).mean()

def decay_imbalance_velocity(df: pd.DataFrame, levels: int = 5, window: int = 10) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        decay_factors = np.exp(-np.arange(levels))
        bid_decay = sum(df[f'bid_qty_{i}'] * decay_factors[i] for i in range(levels))
        ask_decay = sum(df[f'ask_qty_{i}'] * decay_factors[i] for i in range(levels))
        imbalance = (bid_decay - ask_decay) / (bid_decay + ask_decay + 1e-12)
        velocity = imbalance.diff().fillna(0)
        return velocity.rolling(window=window, min_periods=1).mean().fillna(0)
    return pd.Series(0.0, index=df.index)

def liquidity_imbalance_velocity(df: pd.DataFrame, levels: int = 5, window: int = 10) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        imbalance = (sum(df[f'bid_qty_{i}'] for i in range(levels)) - sum(df[f'ask_qty_{i}'] for i in range(levels))) / (sum(df[f'bid_qty_{i}'] + df[f'ask_qty_{i}'] for i in range(levels)) + 1e-12)
        velocity = imbalance.diff().fillna(0)
        return velocity.rolling(window=window, min_periods=1).mean().fillna(0)
    return pd.Series(0.0, index=df.index)

def depth_imbalance_velocity(df: pd.DataFrame, levels: int = 5, window: int = 10) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        imbalance = (sum(df[f'bid_qty_{i}'] for i in range(levels)) - sum(df[f'ask_qty_{i}'] for i in range(levels))) / (sum(df[f'bid_qty_{i}'] + df[f'ask_qty_{i}'] for i in range(levels)) + 1e-12)
        velocity = imbalance.diff().fillna(0)
        return velocity.rolling(window=window, min_periods=1).mean().fillna(0)
    return pd.Series(0.0, index=df.index)

def spread_crossing_event(df: pd.DataFrame) -> pd.Series:
    """Событие кроссинга спреда за 1 бар (цена вышла за пределы спреда)"""
    if 'close' not in df.columns or 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    crossed_up = (df['close'] > df['ask_price_0']).astype(float)
    crossed_down = (df['close'] < df['bid_price_0']).astype(float)
    event = crossed_up - crossed_down
    # Усиление по объёму при кроссинге
    volume_factor = df['volume_60s'] / (df['volume_60s'].rolling(10, min_periods=1).mean() + 1e-12)
    return event * volume_factor.clip(0.5, 3.0)

def price_velocity_1bar(df: pd.DataFrame) -> pd.Series:
    """Скорость цены за 1 бар, нормализованная по спреду"""
    if 'close' not in df.columns or 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    velocity = df['close'].diff().fillna(0)
    spread = df['ask_price_0'] - df['bid_price_0'] + 1e-12
    normalized = velocity / spread
    return normalized * 7.0

def orderbook_tilt_1bar(df: pd.DataFrame) -> pd.Series:
    """Наклон стакана за 1 бар (давление на ближних уровнях)"""
    if 'bid_qty_0' not in df.columns or 'bid_qty_1' not in df.columns or 'ask_qty_0' not in df.columns or 'ask_qty_1' not in df.columns:
        return pd.Series(0.0, index=df.index)
    near_bid = df['bid_qty_0'] + df['bid_qty_1']
    near_ask = df['ask_qty_0'] + df['ask_qty_1']
    tilt = (near_bid - near_ask) / (near_bid + near_ask + 1e-12)
    prev_tilt = tilt.shift(1).fillna(0)
    change = tilt - prev_tilt
    return change * 4.0

def effective_spread_1bar(df: pd.DataFrame) -> pd.Series:
    """Эффективный спред за 1 бар (стоимость транзакции)"""
    if 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns or 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    mid = (df['bid_price_0'] + df['ask_price_0']) / 2
    effective_spread = (df['close'] - mid.shift(1)).abs()
    prev_spread = effective_spread.shift(1).fillna(0)
    change = effective_spread - prev_spread
    normalized = change / (effective_spread + 1e-12)
    return normalized * -4.0

def orderflow_toxicity_1bar(df: pd.DataFrame) -> pd.Series:
    """Токсичность ордерфлоу за 1 бар (соотношение потока к движению)"""
    if 'agg_buy_count_60s' not in df.columns or 'agg_sell_count_60s' not in df.columns or 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    net_flow = df['agg_buy_count_60s'] - df['agg_sell_count_60s']
    price_change = df['close'].diff().fillna(0)
    toxicity = net_flow / (price_change.abs() + 1e-12)
    prev_toxicity = toxicity.shift(1).fillna(0)
    change = toxicity - prev_toxicity
    return change * 0.5

def quote_intensity(df: pd.DataFrame, window: int = 15) -> pd.Series:
    if 'bid_price_0' in df.columns and 'ask_price_0' in df.columns:
        changes = ((df['bid_price_0'].diff() != 0) | (df['ask_price_0'].diff() != 0)).astype(float)
        qi = changes.rolling(window=window, min_periods=1).sum()
        return qi.fillna(0)
    return pd.Series(0.0, index=df.index)

def first_queue_imbalance(df: pd.DataFrame) -> pd.Series:
    if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns:
        fqi = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-12)
        return fqi.fillna(0)
    return pd.Series(0.0, index=df.index)

def quote_fqi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    qi = quote_intensity(df, window)
    fqi = first_queue_imbalance(df)
    return (qi * fqi).fillna(0)

def intensity_fqi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    changes = ((df['bid_qty_0'].diff() != 0) | (df['ask_qty_0'].diff() != 0)).astype(float) if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns else pd.Series(0.0, index=df.index)
    fqi = first_queue_imbalance(df)
    return (changes.rolling(window=window, min_periods=1).mean() * fqi).fillna(0)

def response_speed(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if 'bid_price_0' in df.columns:
        improvements = (df['bid_price_0'].diff() > 0).astype(float)
        rs = improvements.rolling(window=window, min_periods=1).mean()
        return rs.fillna(0)
    return pd.Series(0.0, index=df.index)

def response_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    rs = response_speed(df, window)
    fqi = first_queue_imbalance(df)
    return (rs * fqi).fillna(0)

def exponential_weighted_imbalance(df: pd.DataFrame, levels: int = 10, alpha: float = 0.9) -> pd.Series:
    weights = np.array([alpha ** i for i in range(levels)])
    bid_vol = sum(weights[i] * df[f'bid_qty_{i}'] for i in range(levels) if f'bid_qty_{i}' in df.columns)
    ask_vol = sum(weights[i] * df[f'ask_qty_{i}'] for i in range(levels) if f'ask_qty_{i}' in df.columns)
    ewi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12)
    return ewi.fillna(0)

def depth_ratio(df: pd.DataFrame, levels: int = 5) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        total_bid = sum(df[f'bid_qty_{i}'] for i in range(levels))
        total_ask = sum(df[f'ask_qty_{i}'] for i in range(levels))
        dr = total_bid / (total_ask + 1e-12)
        return np.log(dr + 1).fillna(0)
    return pd.Series(0.0, index=df.index)

def queue_pressure_imbalance(df: pd.DataFrame) -> pd.Series:
    if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns:
        qpi = df['bid_qty_0'] / (df['ask_qty_0'] + 1e-12)
        return np.log(qpi + 1).fillna(0)
    return pd.Series(0.0, index=df.index)

def depth_asymmetry(df: pd.DataFrame, levels: int = 5) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        bid_depth = sum(df[f'bid_qty_{i}'] * (i + 1) for i in range(levels))
        ask_depth = sum(df[f'ask_qty_{i}'] * (i + 1) for i in range(levels))
        da = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
        return da.fillna(0)
    return pd.Series(0.0, index=df.index)

def weighted_depth_ratio(df: pd.DataFrame, levels: int = 10) -> pd.Series:
    weights = np.exp(-np.arange(levels))
    dr = depth_ratio(df, levels)
    wdr = dr * sum(weights)
    return wdr.fillna(0)

def weighted_log_imbalance_5levels(df: pd.DataFrame) -> pd.Series:
    """Взвешенный лог-дисбаланс 5 уровней"""
    weights = np.exp(-np.arange(5) * 0.7)
    bid_log_sum = 0.0
    ask_log_sum = 0.0
    for i, w in enumerate(weights):
        bid = df[f'bid_qty_{i}'].fillna(1e-12).clip(lower=1e-12)
        ask = df[f'ask_qty_{i}'].fillna(1e-12).clip(lower=1e-12)
        bid_log_sum += w * np.log(bid)
        ask_log_sum += w * np.log(ask)
    return bid_log_sum - ask_log_sum

def orderbook_slope_5levels(df: pd.DataFrame) -> pd.Series:
    """Наклон стакана (5 уровней)"""
    bid_sum = 0.0
    ask_sum = 0.0
    weights = [1.0, 0.7, 0.4, 0.2, 0.1]
    for i, w in enumerate(weights):
        if f'bid_qty_{i}' in df.columns:
            bid = df[f'bid_qty_{i}'].fillna(1e-12).clip(lower=1e-12)
            ask = df[f'ask_qty_{i}'].fillna(1e-12).clip(lower=1e-12)
            bid_sum += w * np.log(bid)
            ask_sum += w * np.log(ask)
    return bid_sum - ask_sum

def depth_pressure(df: pd.DataFrame) -> pd.Series:
    eps = 1e-12
    bid_qty = df['bid_qty_0'].fillna(0).clip(lower=eps) if 'bid_qty_0' in df.columns else pd.Series(eps, index=df.index)
    ask_qty = df['ask_qty_0'].fillna(0).clip(lower=eps) if 'ask_qty_0' in df.columns else pd.Series(eps, index=df.index)
    dp = np.log(bid_qty) - np.log(ask_qty)
    return pd.Series(dp, index=df.index)

def order_intensity(df: pd.DataFrame) -> pd.Series:
    """Интенсивность ордеров (частота изменений в стакане)"""
    bid_changed = (df['bid_qty_0'].diff() != 0).astype(float)
    ask_changed = (df['ask_qty_0'].diff() != 0).astype(float)
    return (bid_changed + ask_changed) / 2.0

def order_intensity_weighted(df: pd.DataFrame) -> pd.Series:
    """Взвешенная интенсивность (с учётом дисбаланса)"""
    intensity = order_intensity(df)
    imbalance = depth_pressure(df)
    return intensity * np.tanh(imbalance)

def orderbook_queue_position(df: pd.DataFrame) -> pd.Series:
    """Позиция в очереди (нормализованный объём на первом уровне)"""
    bid_queue = df['bid_qty_0'] / (df['bid_qty_0'].rolling(3, min_periods=1).mean() + 1e-12)
    ask_queue = df['ask_qty_0'] / (df['ask_qty_0'].rolling(3, min_periods=1).mean() + 1e-12)
    return (bid_queue - ask_queue) / (bid_queue + ask_queue + 1e-12)

def microprice(df: pd.DataFrame) -> pd.Series:
    """Классический микропрайс (взвешенный по объёмам)"""
    bid_p = df['bid_price_0'].fillna(0)
    ask_p = df['ask_price_0'].fillna(0)
    bid_v = df['bid_qty_0'].fillna(1e-12)
    ask_v = df['ask_qty_0'].fillna(1e-12)
    total_v = bid_v + ask_v
    return (bid_p * ask_v + ask_p * bid_v) / (total_v + 1e-12)

def microprice_vs_close(df: pd.DataFrame) -> pd.Series:
    """Отклонение микропрайса от цены закрытия"""
    micro = microprice(df)
    spread = df['ask_price_0'] - df['bid_price_0'] + 1e-12
    return (micro - df['close']) / spread

def bid_price_pressure(df: pd.DataFrame) -> pd.Series:
    """Давление на бид (объём × расстояние до следующего уровня)"""
    if 'bid_price_1' in df.columns:
        price_diff = df['bid_price_0'] - df['bid_price_1'].fillna(df['bid_price_0'])
        return df['bid_qty_0'] * price_diff.abs()
    return df['bid_qty_0']

def ask_price_pressure(df: pd.DataFrame) -> pd.Series:
    """Давление на аск"""
    if 'ask_price_1' in df.columns:
        price_diff = df['ask_price_1'].fillna(df['ask_price_0']) - df['ask_price_0']
        return df['ask_qty_0'] * price_diff.abs()
    return df['ask_qty_0']

def price_pressure_imbalance(df: pd.DataFrame) -> pd.Series:
    """Дисбаланс давления на границы"""
    return ask_price_pressure(df) - bid_price_pressure(df)

def quote_intensity_fqi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    qi = quote_intensity(df, window)
    fqi = first_queue_imbalance(df)
    inter = qi * fqi
    return inter.fillna(0)

def strategic_runs(df: pd.DataFrame, window: int = 15) -> pd.Series:
    if 'bid_qty_0' in df.columns:
        changes = (df['bid_qty_0'].diff() != 0).astype(float)
        sr = changes.rolling(window=window, min_periods=1).sum()
        return sr.fillna(0)
    return pd.Series(0.0, index=df.index)

def strategic_fqi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    sr = strategic_runs(df, window)
    fqi = first_queue_imbalance(df)
    inter = sr * fqi
    return inter.fillna(0)

def intensity_fqi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    changes = ((df['bid_qty_0'].diff() != 0) | (df['ask_qty_0'].diff() != 0)).astype(float) if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns else pd.Series(0.0, index=df.index)
    fqi = first_queue_imbalance(df)
    return (changes.rolling(window=window, min_periods=1).mean() * fqi).fillna(0)

def orderbook_queue_position_squared(df: pd.DataFrame) -> pd.Series:
    """Квадрат позиции в очереди"""
    oqp = orderbook_queue_position(df)
    return oqp ** 2 * np.sign(oqp)

def depth_pressure_exp(df: pd.DataFrame) -> pd.Series:
    """Экспоненциальное преобразование дисбаланса — нелинейное усиление"""
    dp = depth_pressure(df)
    return np.sign(dp) * (np.exp(np.abs(dp) * 0.7) - 1)

def weighted_orderbook_imbalance(df, n_levels=10):
    """Weighted imbalance with level decay (exponential weights)."""
    weights = np.exp(-np.arange(n_levels))
    bid_vol = sum(weights[i] * df[f'bid_qty_{i}'] for i in range(n_levels))
    ask_vol = sum(weights[i] * df[f'ask_qty_{i}'] for i in range(n_levels))
    return pd.Series((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12), index=df.index)

def order_intensity_first_queue(df: pd.DataFrame) -> pd.Series:
    """Order arrival intensity with first queue imbalance."""
    changes = ((df['bid_qty_0'].diff() != 0) | (df['ask_qty_0'].diff() != 0)).rolling(15, min_periods=1).mean()
    fqi = first_queue_imbalance(df)
    return pd.Series(changes * fqi, index=df.index)

def orderbook_imbalance(df, n_levels=5):
    bid_volume = sum(df[f'bid_qty_{i}'] for i in range(n_levels)) if 'bid_qty_0' in df.columns else pd.Series(0.0, index=df.index)
    ask_volume = sum(df[f'ask_qty_{i}'] for i in range(n_levels)) if 'ask_qty_0' in df.columns else pd.Series(0.0, index=df.index)
    return pd.Series((bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-12), index=df.index)

def order_intensity_queue_spread_regime(df: pd.DataFrame) -> pd.Series:
    """Order intensity queue in spread regimes with adaptive scaling"""
    intensity_queue = order_intensity_first_queue(df)
    spread = (df['ask_price_0'] - df['bid_price_0']) / ((df['ask_price_0'] + df['bid_price_0']) / 2)
    spread_ma = spread.rolling(15, min_periods=1).mean()
    regime_factor = pd.Series(1.0, index=df.index)
    regime_factor[spread < spread_ma * 0.7] = 1.5
    regime_factor[spread > spread_ma * 1.4] = 0.6
    return pd.Series(intensity_queue * regime_factor, index=df.index)

def log_imbalance_depth_weighted(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns for i in range(n_levels)) and all(f'ask_qty_{i}' in df.columns for i in range(n_levels)):
        bid_depth = sum(df[f'bid_qty_{i}'] * (n_levels - i) for i in range(n_levels))
        ask_depth = sum(df[f'ask_qty_{i}'] * (n_levels - i) for i in range(n_levels))
        imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
        return np.log(imb.abs() + 1) * np.sign(imb)
    return pd.Series(0.0, index=df.index)

def weighted_price_pressure(df: pd.DataFrame, levels: int = 5) -> pd.Series:
    """Взвешенное давление цены с учётом расстояния до мидпрайса"""
    if not all(f'bid_price_{i}' in df.columns and f'ask_price_{i}' in df.columns and f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(levels)):
        return pd.Series(0.0, index=df.index)
    
    mid = (df['bid_price_0'] + df['ask_price_0']) / 2
    bid_pressure = 0.0
    ask_pressure = 0.0
    
    for i in range(levels):
        bid_dist = (mid - df[f'bid_price_{i}']).abs() + 1e-12
        ask_dist = (df[f'ask_price_{i}'] - mid).abs() + 1e-12
        bid_pressure += df[f'bid_qty_{i}'] / bid_dist
        ask_pressure += df[f'ask_qty_{i}'] / ask_dist
    
    net_pressure = bid_pressure - ask_pressure
    total_pressure = bid_pressure + ask_pressure + 1e-12
    return (net_pressure / total_pressure).fillna(0)

def queue_intensity_imbalance(df: pd.DataFrame, window: int = 4) -> pd.Series:
    """Интенсивность изменений очереди × дисбаланс (улучшенный quote_fqi)"""
    if 'bid_qty_0' not in df.columns or 'ask_qty_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    # Интенсивность изменений
    bid_changed = (df['bid_qty_0'].diff().abs() > 0).astype(float)
    ask_changed = (df['ask_qty_0'].diff().abs() > 0).astype(float)
    intensity = (bid_changed + ask_changed).rolling(window=window, min_periods=1).mean()
    
    # Дисбаланс
    fqi = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-12)
    
    return (intensity * fqi).fillna(0)

def spread_regime_imbalance(df: pd.DataFrame, window: int = 6) -> pd.Series:
    """Дисбаланс с адаптацией к режиму спреда"""
    if 'bid_qty_0' not in df.columns or 'ask_qty_0' not in df.columns or 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fqi = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-12)
    spread = df['ask_price_0'] - df['bid_price_0']
    
    # Режим спреда: узкий/широкий относительно скользящего среднего
    spread_ma = spread.rolling(window=window*3, min_periods=1).mean()
    spread_ratio = spread / (spread_ma + 1e-12)
    
    # Усиление в узком спреде (больше информативности)
    regime_factor = np.where(spread_ratio < 0.8, 1.5, np.where(spread_ratio > 1.3, 0.7, 1.0))
    
    return (fqi * regime_factor).fillna(0)

def spread_adjusted_queue_imbalance(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Дисбаланс очереди с поправкой на спред"""
    if 'bid_qty_0' not in df.columns or 'ask_qty_0' not in df.columns or 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fqi = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-12)
    spread = df['ask_price_0'] - df['bid_price_0'] + 1e-12
    
    # В узком спреде дисбаланс информативнее
    spread_ma = spread.rolling(window=window*3, min_periods=1).mean()
    spread_ratio = spread / (spread_ma + 1e-12)
    adjustment = 1.0 / spread_ratio.clip(0.5, 2.0)
    
    return (fqi * adjustment).fillna(0)

def relative_depth_pressure(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    if all(f'bid_qty_{i}' in df.columns and f'ask_qty_{i}' in df.columns for i in range(n_levels)):
        bid_depth = sum(df[f'bid_qty_{i}'] for i in range(n_levels))
        ask_depth = sum(df[f'ask_qty_{i}'] for i in range(n_levels))
        rel = bid_depth / (ask_depth + bid_depth + 1e-12)
        return (2 * rel - 1).fillna(0)
    return pd.Series(0.0, index=df.index)

def f38_fqi_sqrt_times_weighted_imbalance(df):
    a = np.sqrt(np.abs(quote_fqi(df)))
    b = weighted_log_imbalance_5levels(df)
    return (a * b).fillna(0)

def f47_depth_ratio_sign_times_weighted_imbalance(df):
    a = np.sign(depth_ratio(df))
    b = weighted_log_imbalance_5levels(df)
    return (a * b).fillna(0)

def f49_exponential_imbalance_sqrt_times_slope(df):
    a = np.sqrt(np.abs(exponential_weighted_imbalance(df)))
    b = orderbook_slope_5levels(df)
    return (a * b).fillna(0)

def f39_strategic_fqi_abs_times_slope(df):
    a = np.abs(strategic_fqi(df))
    b = orderbook_slope_5levels(df)
    return (a * b).fillna(0)

def combo4_fqi_log1p_times_wli(df):
    fqi = quote_fqi(df)
    wli = weighted_log_imbalance_5levels(df)
    return (np.log1p(np.abs(fqi)) * wli).fillna(0)

def combo6_queue_pressure_sign_times_wli(df):
    qp_sign = np.sign(queue_pressure_imbalance(df))
    wli = weighted_log_imbalance_5levels(df)
    return (qp_sign * wli).fillna(0)

def combo35_fqi_rolling_std_18_times_slope(df):
    fqi_std = quote_fqi(df).rolling(18, min_periods=1).std()
    slope = orderbook_slope_5levels(df)
    return (fqi_std * slope).fillna(0)

def combo5_strategic_fqi_sqrt_times_slope(df):
    sfqi = np.sqrt(np.abs(strategic_fqi(df)))
    slope = orderbook_slope_5levels(df)
    return (sfqi * slope).fillna(0)

def combo7_wli_ewm_span8_diff(df):
    wli = weighted_log_imbalance_5levels(df).ewm(span=8, adjust=False).mean()
    return wli.diff().fillna(0)

def combo16_weighted_orderbook_log_times_slope(df):
    wo_log = np.log1p(np.abs(weighted_orderbook_imbalance(df)))
    slope = orderbook_slope_5levels(df)
    return (wo_log * slope).fillna(0)

def combo29_response_imbalance_log1p_times_wli(df):
    resp_log = np.log1p(np.abs(response_imbalance(df)))
    wli = weighted_log_imbalance_5levels(df)
    return (resp_log * wli).fillna(0)

def combo6_queue_pressure_sign_times_wli(df):
    qp_sign = np.sign(queue_pressure_imbalance(df))
    wli = weighted_log_imbalance_5levels(df)
    return (qp_sign * wli).fillna(0)

def log_imbalance_x_order_intensity(df: pd.DataFrame) -> pd.Series:
    """Лог-дисбаланс × интенсивность ордеров"""
    a = weighted_log_imbalance_5levels(df)
    b = order_intensity(df)
    return (a * np.tanh(b * 3.0) * 1.3).fillna(0)

def slope_x_intensity(df: pd.DataFrame) -> pd.Series:
    """Наклон стакана × интенсивность изменений (режимный сигнал)"""
    a = orderbook_slope_5levels(df)
    b = quote_intensity(df, window=5)  # короткое окно для 5с
    return (a * np.tanh(b * 0.3)).fillna(0)

def log_imbalance_x_spread_regime(df: pd.DataFrame) -> pd.Series:
    """Лог-дисбаланс × режим спреда (адаптивное усиление)"""
    a = weighted_log_imbalance_5levels(df)
    spread = df['ask_price_0'] - df['bid_price_0'] + 1e-12
    spread_ma = spread.rolling(12, min_periods=1).mean()
    regime = np.where(spread < spread_ma * 0.75, 2.0, np.where(spread > spread_ma * 1.4, 0.4, 1.0))
    return (a * regime).fillna(0)

def top_analog24_slope_abs_log1p_power2_times_wli_sign(df):
    slope_abs_log_p2 = np.power(np.log1p(np.abs(orderbook_slope_5levels(df))), 2)
    wli_sign = np.sign(weighted_log_imbalance_5levels(df))
    return (slope_abs_log_p2 * wli_sign).fillna(0)








def keltner_channel(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel width: (upper - lower) / middle period=20"""
    upper = df['close'].ewm(span=20).mean() + 2 * talib.ATR(df['high'], df['low'], df['close'], 20)
    lower = df['close'].ewm(span=20).mean() - 2 * talib.ATR(df['high'], df['low'], df['close'], 20)
    middle = df['close'].ewm(span=20).mean()
    return (upper - lower) / middle

def hl_range_normalization_from_book(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        hl = (df['high'] - df['low']) / df['close']
        norm = hl.rolling(window=window, min_periods=1).mean()
        return norm.fillna(0)
    return pd.Series(0.0, index=df.index)

def implied_volatility_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if 'high' in df.columns and 'low' in df.columns:
        hl = (df['high'] - df['low']) / df['close']
        iv = hl.rolling(window=window, min_periods=1).mean() * np.sqrt(252)
        return iv.fillna(0)
    return pd.Series(0.0, index=df.index)

def atr(df: pd.DataFrame) -> pd.Series:
    return talib.ATR(df['high'], df['low'], df['close'], timeperiod=30)

def realized_range_volatility(df: pd.DataFrame, window: int = 15) -> pd.Series:
    """Волатильность на основе реального диапазона (не только закрытия)"""
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    log_hc = np.log(df['high'] / df['close'])
    log_lc = np.log(df['low'] / df['close'])
    range_vol = np.sqrt((log_hc**2 + log_lc**2).rolling(window, min_periods=1).mean())
    return range_vol * np.sqrt(window)

def order_flow_symmetry_index(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Индекс симметрии ордерфлоу — баланс агрессивных покупок и продаж"""
    if 'agg_buy_count_60s' not in df.columns or 'agg_sell_count_60s' not in df.columns:
        return pd.Series(0.0, index=df.index)
    buy_flow = df['agg_buy_count_60s'].rolling(window, min_periods=1).sum()
    sell_flow = df['agg_sell_count_60s'].rolling(window, min_periods=1).sum()
    symmetry = 1 - (buy_flow - sell_flow).abs() / (buy_flow + sell_flow + 1e-12)
    flow_stability = 1 - ((buy_flow.diff().fillna(0).abs() + sell_flow.diff().fillna(0).abs()) / 
                         (buy_flow + sell_flow + 1e-12)).rolling(5, min_periods=1).mean()
    return symmetry * flow_stability * (df['high'] - df['low']).rolling(window, min_periods=1).mean()

def parkinson_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if 'high' in df.columns and 'low' in df.columns:
        log_hl = np.log(df['high'] / df['low']) ** 2
        pv = np.sqrt(log_hl.rolling(window=window, min_periods=1).sum() / (4 * np.log(2) * window))
        return pd.Series(pv, index=df.index).fillna(0)
    else:
        return pd.Series(0.0, index=df.index)
    
def high_low_normalized(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        hl_range = (df['high'] - df['low']) / df['close']
        return hl_range.rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        return pd.Series(0.0, index=df.index)
    
def garman_klass_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        gk = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        return gk.rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        return pd.Series(0.0, index=df.index)
    
def rogers_satchell_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
        log_hc = np.log(df['high'] / df['close'])
        log_ho = np.log(df['high'] / df['open'])
        log_lc = np.log(df['low'] / df['close'])
        log_lo = np.log(df['low'] / df['open'])
        rs = log_hc * log_ho + log_lc * log_lo
        return np.sqrt(rs.rolling(window=window, min_periods=1).mean()).fillna(0)
    else:
        return pd.Series(0.0, index=df.index)
    
def herding_measure(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if 'close' in df.columns:
        returns = df['close'].pct_change().fillna(0)
        herd = np.abs(returns - returns.mean()) / returns.std()
        measure = herd.rolling(window=window, min_periods=1).mean()
        return measure.fillna(0)
    return pd.Series(0.0, index=df.index)

def realized_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if 'close' in df.columns:
        returns = df['close'].pct_change().fillna(0)
        rv = np.sqrt((returns ** 2).rolling(window=window, min_periods=1).sum())
        return rv.fillna(0)
    return pd.Series(0.0, index=df.index)
    
def volatility_contraction_expansion_ratio(df: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> pd.Series:
    """Low ratio indicating contraction"""
    short_vol = realized_volatility(df, window=short_window)
    long_vol = realized_volatility(df, window=long_window)
    ratio = short_vol / (long_vol + 1e-12)
    contraction = (ratio < 1).astype(float)
    return contraction * herding_measure(df)

def keltner_channel_width_from_mid(df: pd.DataFrame, window: int = 20) -> pd.Series:
    mid = (df['bid_price_0'] + df['ask_price_0']) / 2
    ma = mid.ewm(span=window, min_periods=1).mean()
    atr_val = (mid.diff().abs().rolling(window=window, min_periods=1).mean()) * np.sqrt(252)
    upper = ma + 2 * atr_val
    lower = ma - 2 * atr_val
    width = (upper - lower) / ma
    return width.fillna(0)

def adverse_selection_risk(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if 'close' in df.columns:
        mid = (df['bid_price_0'] + df['ask_price_0']) / 2 if 'bid_price_0' in df.columns else df['close']
        price_change = mid.diff().fillna(0)
        adv = price_change.abs().rolling(window=window, min_periods=1).mean()
        return adv.fillna(0)
    return pd.Series(0.0, index=df.index)

def bid_ask_spread_tightness_index(df: pd.DataFrame, window: int = 15) -> pd.Series:
    """Индекс стянутости спреда - узкий спред для флетового рынка"""
    if 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    spread = df['ask_price_0'] - df['bid_price_0']
    normalized_spread = spread / ((df['bid_price_0'] + df['ask_price_0']) / 2)
    tightness = 1 - normalized_spread.rolling(window, min_periods=1).mean()
    return tightness * adverse_selection_risk(df)

def double_touch_bid_ask_price_balance(df: pd.DataFrame) -> pd.Series:
    """Баланс цен bid/ask за 5 секунд"""
    if 'bid_price_0' not in df.columns or 'ask_price_0' not in df.columns:
        return pd.Series(0.0, index=df.index)
    mid_price = (df['bid_price_0'] + df['ask_price_0']) / 2
    bid_dev = (mid_price - df['bid_price_0']) / mid_price
    ask_dev = (df['ask_price_0'] - mid_price) / mid_price
    balance = 1 - (bid_dev - ask_dev).abs() / (bid_dev + ask_dev + 1e-12)
    return balance * adverse_selection_risk(df)

def spread_ema_10(df: pd.DataFrame) -> pd.Series:
    """Экспоненциальное среднее спреда со средним окном"""
    spread = df['high'] - df['low']
    return spread.ewm(span=10, min_periods=1).mean().fillna(0)

def spread_log_transform(df: pd.DataFrame) -> pd.Series:
    """Логарифм спреда — усиливает чувствительность к малым изменениям"""
    spread = df['high'] - df['low']
    return np.log1p(spread).rolling(window=5, min_periods=1).mean().fillna(0)

def spread_95_percentile_15(df: pd.DataFrame) -> pd.Series:
    """95-й перцентиль спреда за 15 периодов — предиктор расширения"""
    spread = df['high'] - df['low']
    return spread.rolling(window=15, min_periods=1).quantile(0.95).fillna(0)

def spread_squared_ema(df: pd.DataFrame) -> pd.Series:
    """Квадрат спреда с экспоненциальным сглаживанием — фокус на экстремумы"""
    spread = df['high'] - df['low']
    return (spread ** 2).ewm(span=3, min_periods=1).mean().fillna(0)

def spread_weighted_mean_10(df: pd.DataFrame) -> pd.Series:
    """Взвешенное среднее спреда за 10 периодов (экспоненциальные веса)"""
    spread = df['high'] - df['low']
    return spread.ewm(span=10, min_periods=1).mean().fillna(0)

def spread_ema_8(df: pd.DataFrame) -> pd.Series:
    """Экспоненциальное среднее спреда с окном 8 — компромисс между коротким и средним"""
    spread = df['high'] - df['low']
    return spread.ewm(span=8, min_periods=1).mean().fillna(0)

def spread_volume_regime_ema(df: pd.DataFrame) -> pd.Series:
    """EMA спреда с адаптацией под объемный режим"""
    spread = df['high'] - df['low']
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    high_vol_regime = (volume > volume.rolling(window=15, min_periods=1).mean()).astype(float)
    # При высоком объеме — короткое окно (реакция на расширение), при низком — длинное
    ema_short = spread.ewm(span=6, min_periods=1).mean()
    ema_long = spread.ewm(span=18, min_periods=1).mean()
    adaptive = high_vol_regime * ema_short + (1 - high_vol_regime) * ema_long
    return adaptive.fillna(0)

def spread_regime_expansion_ema(df: pd.DataFrame) -> pd.Series:
    """EMA спреда с переключением режимов: короткое окно при расширении, длинное при сжатии"""
    spread = df['high'] - df['low']
    short_ema = spread.ewm(span=5, min_periods=1).mean()
    long_ema = spread.ewm(span=20, min_periods=1).mean()
    regime = (spread > spread.rolling(window=15, min_periods=1).median()).astype(float)
    adaptive_ema = regime * short_ema + (1 - regime) * long_ema
    return adaptive_ema.fillna(0)

def spread_regime_expansion_ema_v2(df: pd.DataFrame) -> pd.Series:
    """Улучшенный режимный EMA: три режима (сжатие/нейтральный/расширение)"""
    spread = df['high'] - df['low']
    ema_5 = spread.ewm(span=5, min_periods=1).mean()
    ema_10 = spread.ewm(span=10, min_periods=1).mean()
    ema_20 = spread.ewm(span=20, min_periods=1).mean()
    
    q25 = spread.rolling(window=20, min_periods=1).quantile(0.25)
    q75 = spread.rolling(window=20, min_periods=1).quantile(0.75)
    
    regime = pd.Series(1.0, index=df.index)  # нейтральный по умолчанию
    regime[spread < q25] = 0.5  # сжатие → длинное окно
    regime[spread > q75] = 2.0  # расширение → короткое окно
    
    adaptive = regime * ema_5 + (2 - regime) * ema_20
    return adaptive.fillna(0)

def spread_agg_flow_weighted_ema_12(df: pd.DataFrame) -> pd.Series:
    """Спред взвешенный по агрессивному ордерфлоу"""
    spread = df['high'] - df['low']
    if 'agg_buy_count_60s' in df.columns and 'agg_sell_count_60s' in df.columns:
        agg_flow = df['agg_buy_count_60s'] + df['agg_sell_count_60s']
        vw_spread = (spread * agg_flow).ewm(span=12, min_periods=1).mean() / agg_flow.ewm(span=12, min_periods=1).mean()
        return vw_spread.fillna(0)
    return spread.ewm(span=12, min_periods=1).mean().fillna(0)

def atr_exponential_smoothing_alpha_0_8(df: pd.DataFrame) -> pd.Series:
    """Экспоненциальное сглаживание ATR с альфа=0.8 — очень реактивное"""
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return atr.ewm(alpha=0.8, min_periods=1).mean().fillna(0)

def hl_vol_ewm_mean(df: pd.DataFrame, span: int = 15) -> pd.Series:
    hlv = parkinson_volatility(df, window=1)
    return hlv.ewm(span=span, min_periods=1).mean().fillna(0)

def microprice_sensitivity(df: pd.DataFrame) -> pd.Series:
    """Чувствительность микроприцы к изменениям стакана"""
    if 'bid_price_0' in df.columns and 'ask_price_0' in df.columns and 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns:
        microprice = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'])
        sensitivity = microprice.diff().abs() / ((df['ask_price_0'] - df['bid_price_0']) + 1e-12)
        return sensitivity.rolling(window=10, min_periods=1).mean()
    return pd.Series(0.0, index=df.index)

def spread_nonlinear_reversion_regime(df: pd.DataFrame) -> pd.Series:
    """Режим нелинейного возврата спреда к среднему"""
    spread = df['high'] - df['low']
    mean = spread.rolling(window=22, min_periods=1).mean()
    deviation = spread - mean
    reversion_force = -np.sign(deviation) * np.abs(deviation) ** 1.3
    regime_strength = reversion_force.abs().rolling(window=10, min_periods=1).mean()
    return regime_strength.fillna(0)

def order_flow_imbalance_nonlinear2(df: pd.DataFrame) -> pd.Series:
    ofsi = order_flow_symmetry_index(df, window=20)
    pv = parkinson_volatility(df, window=25)
    return ofsi * np.sqrt(pv ** 2) * ofsi.diff().fillna(0).rolling(window=20, min_periods=1).std().fillna(0)

def bid_ask_cancellation_proxy5(df: pd.DataFrame) -> pd.Series:
    ofsi = order_flow_symmetry_index(df, window=10)
    rv = realized_volatility(df, window=15)
    return np.exp(1 - ofsi.abs()) * rv * ofsi.rolling(window=10, min_periods=1).mean().fillna(0)

def spread_percentile_5_8(df: pd.DataFrame) -> pd.Series:
    """5-й перцентиль спреда за 8 периодов — предиктор сжатия"""
    spread = df['high'] - df['low']
    return spread.rolling(window=8, min_periods=1).quantile(0.05).fillna(0)

def spread_percentile_95_8(df: pd.DataFrame) -> pd.Series:
    """95-й перцентиль спреда за 8 периодов — предиктор экстремального расширения"""
    spread = df['high'] - df['low']
    return spread.rolling(window=8, min_periods=1).quantile(0.95).fillna(0)

def spread_ema_15(df: pd.DataFrame) -> pd.Series:
    """Длинное EMA спреда — трендовый уровень"""
    spread = df['high'] - df['low']
    return spread.ewm(span=15, min_periods=1).mean().fillna(0)