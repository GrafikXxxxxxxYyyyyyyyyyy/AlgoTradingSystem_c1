import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from itertools import product
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from src.mlcore.dataloader import calculate_features



class BaseTrader(ABC):
    """
    Базовый класс для всех трейдеров с общей логикой управления позицией и ордерами
    """
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: float,
        max_position: float = 0.01,
        position_qty: float = 0.002,
    ):
        # Параметры торговли
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_position = max_position
        self.position_qty = position_qty

        # Состояние позиции (единая логика для всех трейдеров)
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0

        # Текущие ордера
        self.buy_order: Optional[Dict[str, Any]] = None
        self.sell_order: Optional[Dict[str, Any]] = None

    def calculate_optimal_quotes(
        self,
        last_price: float,
        mid_pred: float,
        spread_pred: float,
        min_spread_bps: float = 1.0,
        vol_widen_bps: float = 0.0,
        atr_or_vol: float = 0.0,
    ):
        """
        min_spread_bps: минимальная половинка спреда в б.п. (защита от слишком плотных котировок).
        vol_widen_bps: множитель волатильности для расширения спреда (0 = отключено).
        atr_or_vol: значение ATR или волатильности (в тех же единицах что last_price) для расширения.
        """
        optimal_spread = self.alpha * spread_pred * last_price
        directional_drift = self.beta * last_price * (np.exp(mid_pred) - 1)
        positional_skew = self.gamma * (self.position / self.max_position) * spread_pred * last_price

        if vol_widen_bps > 0 and atr_or_vol > 0:
            optimal_spread = optimal_spread + (vol_widen_bps * 1e-4 * last_price * atr_or_vol)

        half_spread = max(optimal_spread, last_price * (min_spread_bps * 1e-4))

        bid_adjustment = 0.0
        ask_adjustment = 0.0
        if self.avg_entry_price != 0 and self.position != 0:
            if self.position > 0:
                ask_adjustment = self.epsilon * (last_price - self.avg_entry_price)
            elif self.position < 0:
                bid_adjustment = self.epsilon * (last_price - self.avg_entry_price)

        bid_price = last_price - half_spread + directional_drift - positional_skew + bid_adjustment
        ask_price = last_price + half_spread + directional_drift - positional_skew + ask_adjustment

        bid_price = min(last_price - 0.1, bid_price)
        ask_price = max(last_price + 0.1, ask_price)
        if ask_price <= bid_price:
            ask_price = bid_price + max(0.1, last_price * (min_spread_bps * 1e-4))

        return bid_price, ask_price

    def handle_buy_order(
        self,
        fill_price: Optional[float] = None,
        commission: float = 0.0,
        actual_filled_qty: Optional[float] = None,
    ) -> None:
        """actual_filled_qty: фактически исполненный объём (при частичном исполнении). Если None — берётся origQty."""
        qty = float(actual_filled_qty if actual_filled_qty is not None else self.buy_order['origQty'])
        if qty <= 0:
            self.buy_order = None
            return
        order_price = float(self.buy_order['price'])
        if fill_price is None:
            fill_price = order_price

        if self.position >= 0:
            new_position = self.position + qty
            self.avg_entry_price = (
                (self.position * self.avg_entry_price + qty * fill_price) / new_position
            ) if new_position != 0 else fill_price
            self.position = new_position
        else:
            qty_to_close = min(abs(self.position), qty)
            self.realized_pnl += qty_to_close * (self.avg_entry_price - fill_price)
            self.position += qty_to_close
            
            remaining_qty = qty - qty_to_close
            if remaining_qty > 0:
                self.position = remaining_qty
                self.avg_entry_price = fill_price
            elif self.position == 0.0:
                self.avg_entry_price = 0.0
        
        self.realized_pnl -= commission
        self.buy_order = None

    def handle_sell_order(
        self,
        fill_price: Optional[float] = None,
        commission: float = 0.0,
        actual_filled_qty: Optional[float] = None,
    ) -> None:
        """actual_filled_qty: фактически исполненный объём (при частичном исполнении). Если None — берётся origQty."""
        qty = float(actual_filled_qty if actual_filled_qty is not None else self.sell_order['origQty'])
        if qty <= 0:
            self.sell_order = None
            return
        order_price = float(self.sell_order['price'])
        if fill_price is None:
            fill_price = order_price

        if self.position <= 0:
            new_position = self.position - qty
            self.avg_entry_price = (
                (self.position * self.avg_entry_price + (-qty) * fill_price) / new_position
            ) if new_position != 0 else fill_price
            self.position = new_position
        else:
            qty_to_close = min(self.position, qty)
            self.realized_pnl += qty_to_close * (fill_price - self.avg_entry_price)
            self.position -= qty_to_close
            
            remaining_qty = qty - qty_to_close
            if remaining_qty > 0:
                self.position = -remaining_qty
                self.avg_entry_price = fill_price
            elif self.position == 0.0:
                self.avg_entry_price = 0.0

        self.realized_pnl -= commission
        self.sell_order = None

    @abstractmethod
    async def background_monitoring(self) -> None:
        """Логирование результатов и мониторинг исполнения ордеров (специфично для среды)"""
        pass

    @abstractmethod
    async def cancel_old_orders(self) -> None:
        """Отмена существующих ордеров (специфично для среды)"""
        pass

    @abstractmethod
    async def place_new_orders(self, last_price: float, mid_pred: float, spread_pred: float) -> None:
        """Выставление новых ордеров (специфично для среды)"""
        pass

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        """Основной торговый цикл (точка входа)"""
        pass



class BacktestTrader(BaseTrader):
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: float,
        max_position: float = 0.01,
        position_qty: float = 0.002,
        commission_bps: float = 2.0,
        slippage_bps: float = 0.0,
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            max_position=max_position,
            position_qty=position_qty
        )
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

        self.position_curve = []
        self.realized_pnl_curve = []
        self.unrealized_pnl_curve = []
        self.total_pnl_curve = []
        self.n_trades = 0
        self.total_commission = 0.0

    def _fill_price_and_commission(self, order_price: float, qty: float, side: str) -> Tuple[float, float]:
        """Реалистичное исполнение: проскальзывание (хуже для нас) и комиссия."""
        bps = 1e-4
        if side == "buy":
            fill_price = order_price * (1 + self.slippage_bps * bps)
        else:
            fill_price = order_price * (1 - self.slippage_bps * bps)
        commission = qty * fill_price * (self.commission_bps * bps)
        return fill_price, commission

    def background_monitoring(self, low, high, close):
        if self.buy_order and low <= self.buy_order['price']:
            qty = float(self.buy_order['origQty'])
            order_price = float(self.buy_order['price'])
            fill_price, commission = self._fill_price_and_commission(order_price, qty, "buy")
            self.total_commission += commission
            self.n_trades += 1
            self.handle_buy_order(fill_price=fill_price, commission=commission)

        if self.sell_order and high >= self.sell_order['price']:
            qty = float(self.sell_order['origQty'])
            order_price = float(self.sell_order['price'])
            fill_price, commission = self._fill_price_and_commission(order_price, qty, "sell")
            self.total_commission += commission
            self.n_trades += 1
            self.handle_sell_order(fill_price=fill_price, commission=commission)

        if self.position != 0.0:
            unrealized = self.position * (close - self.avg_entry_price)
        else:
            unrealized = 0.0

        total_pnl = self.realized_pnl + unrealized

        self.position_curve.append(self.position)
        self.realized_pnl_curve.append(self.realized_pnl)
        self.unrealized_pnl_curve.append(unrealized)
        self.total_pnl_curve.append(total_pnl)

    def cancel_old_orders(self):
        self.buy_order = None
        self.sell_order = None

    def place_new_orders(self, last_price, mid_pred, spread_pred):
        buy_price, sell_price = self.calculate_optimal_quotes(
            last_price=last_price,
            mid_pred=mid_pred,
            spread_pred=spread_pred,
        )

        can_buy = self.position + self.position_qty <= self.max_position
        can_sell = self.position - self.position_qty >= -self.max_position

        if can_buy:
            self.buy_order = {
                'price': buy_price,
                'origQty': self.position_qty,
            }

        if can_sell:
            self.sell_order = {
                'price': sell_price,
                'origQty': self.position_qty,
            }

    def run(
        self,
        ohlcv: pd.DataFrame,
        mid_predicts: np.ndarray,
        spread_predicts: np.ndarray,
        position_size: float = 100.0,
        bar_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """
        bar_seconds : длительность одного бара в секундах (для годовой нормализации Sharpe).
                      5.0 для 5s, 1.0 для 1s, 0.1 для 100ms.
        """
        assert len(ohlcv) == len(mid_predicts)
        assert len(ohlcv) == len(spread_predicts)
        n = len(ohlcv)

        # Buy & Hold baseline
        initial_price = ohlcv['close'].iloc[0]
        bh_pnl = (ohlcv['close'] / initial_price - 1) * position_size
        bh_curve = bh_pnl.tolist()

        for i in range(n):
            low = ohlcv['low'].iloc[i]
            high = ohlcv['high'].iloc[i]
            close = ohlcv['close'].iloc[i]

            # 1. Проверяем исполнение ордеров (с комиссией и проскальзыванием)
            self.background_monitoring(low, high, close)
            # 2. Отменяем старые ордеры
            self.cancel_old_orders()
            # 3. Выставляем новые ордеры
            self.place_new_orders(close, mid_predicts[i], spread_predicts[i])

        # Метрики
        pnl_series = pd.Series(self.total_pnl_curve)
        returns = pnl_series.diff().dropna() / (position_size + 1e-12)
        periods_per_year = (365 * 24 * 60 * 60) / max(bar_seconds, 0.1)

        sharpe_ratio = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

        # Максимальная просадка по кривой total PnL
        cum = np.array(self.total_pnl_curve, dtype=float)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

        gross_pnl = self.realized_pnl + (self.position * (ohlcv['close'].iloc[-1] - self.avg_entry_price) if self.position != 0 else 0.0)
        net_pnl = gross_pnl - self.total_commission
        final_total = self.total_pnl_curve[-1] if self.total_pnl_curve else 0.0

        results = {
            'bh_curve': bh_curve,
            'position_curve': self.position_curve,
            'realized_pnl_curve': self.realized_pnl_curve,
            'unrealized_pnl_curve': self.unrealized_pnl_curve,
            'total_pnl_curve': self.total_pnl_curve,
            'sharpe_ratio': sharpe_ratio,
            'final_pnl': final_total,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_commission': self.total_commission,
            'n_trades': self.n_trades,
            'max_drawdown': max_drawdown,
        }
        return results
    


def optimize_parameters(
    ohlcv: pd.DataFrame,
    mid_predicts: np.ndarray,
    spread_predicts: np.ndarray,
    position_size: float = 100.0,
    alpha_range: Tuple[float, float, float] = (0.0, 10.0, 0.1),
    beta_range: Tuple[float, float, float] = (0.0, 10.0, 0.1),
    gamma_range: Tuple[float, float, float] = (0.0, 10.0, 0.1),
    epsilon_range: Tuple[float, float, float] = (0.0, 10.0, 0.1),
    max_position: float = 0.01,
    position_qty: float = 0.002,
) -> Dict[str, any]:
    """
    Оптимизация параметров alpha, beta, gamma с помощью grid search для максимизации коэффициента Шарпа.
    """
    best_sharpe = -np.inf
    best_params = None
    best_results = None
    
    alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2])
    betas = np.arange(beta_range[0], beta_range[1] + beta_range[2], beta_range[2])
    gammas = np.arange(gamma_range[0], gamma_range[1] + gamma_range[2], gamma_range[2])
    epsilons = np.arange(epsilon_range[0], epsilon_range[1] + epsilon_range[2], epsilon_range[2])
    
    total_combinations = len(alphas) * len(betas) * len(gammas) * len(epsilons)
    print(f"Оптимизация: {total_combinations} комбинаций параметров...")
    
    for alpha, beta, gamma, epsilon in tqdm(product(alphas, betas, gammas, epsilons)):
        backtester = BacktestTrader(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            max_position=max_position,
            position_qty=position_qty,
        )
        
        results = backtester.run(
            ohlcv=ohlcv,
            mid_predicts=mid_predicts,
            spread_predicts=spread_predicts,
            position_size=position_size,
        )
        
        sharpe = results['sharpe_ratio']
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'epsilon': epsilon}
            best_results = results
    
    print(f"Лучшие параметры: {best_params}")
    print(f"Максимальный Sharpe: {best_sharpe}")
    
    return {
        'best_params': best_params,
        'best_sharpe': best_sharpe,
        'best_results': best_results
    }



def objective(params, ohlcv, mid_predicts, spread_predicts, position_size, max_position, position_qty):
    alpha, beta, gamma, epsilon = params
    backtester = BacktestTrader(
        alpha=alpha, beta=beta, gamma=gamma, epsilon=epsilon,
        max_position=max_position, position_qty=position_qty
    )
    results = backtester.run(
        ohlcv=ohlcv,
        mid_predicts=mid_predicts,
        spread_predicts=spread_predicts,
        position_size=position_size,
    )
    # Минимизируем отрицательный Sharpe (gp_minimize ищет минимум)
    return -results['sharpe_ratio']


def optimize_parameters_fast(
    ohlcv: pd.DataFrame,
    mid_predicts: np.ndarray,
    spread_predicts: np.ndarray,
    position_size: float = 100.0,
    max_position: float = 0.02,
    position_qty: float = 0.002,
    alpha_range: Tuple[float, float] = (0.0, 10.0),
    beta_range: Tuple[float, float] = (0.0, 10.0),
    gamma_range: Tuple[float, float] = (0.0, 10.0),
    epsilon_range: Tuple[float, float] = (0.0, 10.0),
    n_calls: int = 300,
    n_initial_points: int = 50,
):
    try:
        from skopt.space import Real
        from skopt import gp_minimize
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Для optimize_parameters_fast нужен scikit-optimize. Установите: pip install scikit-optimize"
        ) from e

    # Определяем пространство поиска (сужаем на основе практики)
    space = [
        Real(alpha_range[0], alpha_range[1], name='alpha'),   # Spread scaling
        Real(beta_range[0], beta_range[1], name='beta'),    # Directional component (часто мал)
        Real(gamma_range[0], gamma_range[1], name='gamma'),   # Positional skew
        Real(epsilon_range[0], epsilon_range[1], name='epsilon')  # PnL skew (часто мал)
    ]

    # Фиксируем остальные параметры через partial
    objective_partial = partial(
        objective,
        ohlcv=ohlcv,
        mid_predicts=mid_predicts,
        spread_predicts=spread_predicts,
        position_size=position_size,
        max_position=max_position,
        position_qty=position_qty
    )

    # Bayesian optimization
    result = gp_minimize(
        objective_partial,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=True
    )
    
    best_params = {
        'alpha': result.x[0],
        'beta': result.x[1],
        'gamma': result.x[2],
        'epsilon': result.x[3]
    }
    
    # Финальный бэктест с лучшими параметрами
    final_backtester = BacktestTrader(**best_params, max_position=max_position, position_qty=position_qty)
    best_results = final_backtester.run(
        ohlcv=ohlcv,
        mid_predicts=mid_predicts,
        spread_predicts=spread_predicts,
        position_size=position_size,
    )
    
    return {
        'best_params': best_params,
        'best_sharpe': -result.fun,  # Возвращаем положительный Sharpe
        'best_results': best_results,
        'optimization_history': result  # Для анализа сходимости
    }