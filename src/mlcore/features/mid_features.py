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