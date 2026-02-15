import talib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf, erfinv, gammaln, betainc, beta, gamma, digamma



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