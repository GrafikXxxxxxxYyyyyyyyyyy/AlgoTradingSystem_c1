import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List
from src.trading.mm_backtest_numba import numba_available, simulate_mm_backtest
from src.trading.nsga2_pareto import run_nsga2



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

        # Последние цены входа по стороне (для realized PnL при сверке с пост-состоянием биржи после флипа)
        self._last_long_entry: float = 0.0
        self._last_short_entry: float = 0.0

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
        exchange_post: Optional[Tuple[float, float]] = None,
    ) -> None:
        """actual_filled_qty: фактически исполненный объём (при частичном исполнении). Если None — берётся origQty."""
        qty = float(actual_filled_qty if actual_filled_qty is not None else self.buy_order['origQty'])
        if qty <= 0:
            self.buy_order = None
            return
        order_price = float(self.buy_order['price'])
        if fill_price is None:
            fill_price = order_price

        # Актуальное (post-fill) состояние с биржи: иначе локальное «до сделки» могло быть затёрто sync.
        if exchange_post is not None:
            post_P, post_E = exchange_post
            p = float(fill_price)
            q = qty
            pre_P = post_P - q
            delta = 0.0
            if pre_P < 0:
                closed_short = min(abs(pre_P), q)
                if post_P < 0:
                    entry_ref = post_E
                else:
                    entry_ref = self._last_short_entry
                delta = closed_short * (entry_ref - p)
            self.realized_pnl += delta - commission
            self.position = post_P
            self.avg_entry_price = post_E
            if post_P > 0:
                self._last_long_entry = post_E
            elif post_P < 0:
                self._last_short_entry = post_E
            self.buy_order = None
            return

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
        exchange_post: Optional[Tuple[float, float]] = None,
    ) -> None:
        """actual_filled_qty: фактически исполненный объём (при частичном исполнении). Если None — берётся origQty."""
        qty = float(actual_filled_qty if actual_filled_qty is not None else self.sell_order['origQty'])
        if qty <= 0:
            self.sell_order = None
            return
        order_price = float(self.sell_order['price'])
        if fill_price is None:
            fill_price = order_price

        if exchange_post is not None:
            post_P, post_E = exchange_post
            p = float(fill_price)
            q = qty
            pre_P = post_P + q
            delta = 0.0
            if pre_P > 0:
                closed_long = min(pre_P, q)
                if post_P > 0:
                    entry_ref = post_E
                else:
                    entry_ref = self._last_long_entry
                delta = closed_long * (p - entry_ref)
            self.realized_pnl += delta - commission
            self.position = post_P
            self.avg_entry_price = post_E
            if post_P > 0:
                self._last_long_entry = post_E
            elif post_P < 0:
                self._last_short_entry = post_E
            self.sell_order = None
            return

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
        stride: int = 1,
        use_numba: Optional[bool] = None,
        store_curves: bool = True,
    ) -> Dict[str, Any]:
        """
        bar_seconds : длительность одного бара в секундах (для годовой нормализации Sharpe).
        stride : использовать каждый stride-й бар (увеличивает скорость, меняет метрики; для оптимизации обычно 1 или >1 без store_curves).
        use_numba : None — использовать Numba, если установлен и доступен симулятор.
        store_curves : при False не строятся кривые (ускорение и экономия памяти).
        """
        assert len(ohlcv) == len(mid_predicts)
        assert len(ohlcv) == len(spread_predicts)
        n_all = len(ohlcv)
        assert stride >= 1

        close_np = np.ascontiguousarray(ohlcv["close"].to_numpy(dtype=np.float64, copy=False))
        initial_price = float(close_np[0])
        bh_pnl = (close_np / initial_price - 1.0) * position_size
        bh_curve = bh_pnl.tolist()

        if use_numba is None:
            use_numba = numba_available()

        low_np = np.ascontiguousarray(ohlcv["low"].to_numpy(dtype=np.float64, copy=False))
        high_np = np.ascontiguousarray(ohlcv["high"].to_numpy(dtype=np.float64, copy=False))
        mid_np = np.ascontiguousarray(np.asarray(mid_predicts, dtype=np.float64))
        spread_np = np.ascontiguousarray(np.asarray(spread_predicts, dtype=np.float64))

        n_eff = (n_all + stride - 1) // stride
        curves_empty: List[float] = []

        if use_numba and numba_available():
            if store_curves:
                pc = np.empty(n_eff, dtype=np.float64)
                rc = np.empty(n_eff, dtype=np.float64)
                uc = np.empty(n_eff, dtype=np.float64)
                tp = np.empty(n_eff, dtype=np.float64)
            else:
                pc = rc = uc = tp = np.empty(0, dtype=np.float64)

            (
                sharpe_ratio,
                final_total,
                gross_pnl,
                net_pnl,
                max_drawdown,
                n_trades,
                total_commission,
                n_out,
            ) = simulate_mm_backtest(
                low_np,
                high_np,
                close_np,
                mid_np,
                spread_np,
                float(self.alpha),
                float(self.beta),
                float(self.gamma),
                float(self.epsilon),
                float(self.max_position),
                float(self.position_qty),
                float(self.commission_bps),
                float(self.slippage_bps),
                float(position_size),
                float(bar_seconds),
                1.0,
                int(stride),
                bool(store_curves),
                pc,
                rc,
                uc,
                tp,
            )

            if store_curves and n_out > 0:
                results = {
                    "bh_curve": bh_curve,
                    "position_curve": pc[: int(n_out)].tolist(),
                    "realized_pnl_curve": rc[: int(n_out)].tolist(),
                    "unrealized_pnl_curve": uc[: int(n_out)].tolist(),
                    "total_pnl_curve": tp[: int(n_out)].tolist(),
                    "sharpe_ratio": float(sharpe_ratio),
                    "final_pnl": float(final_total),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "total_commission": float(total_commission),
                    "n_trades": int(n_trades),
                    "max_drawdown": float(max_drawdown),
                }
            else:
                results = {
                    "bh_curve": bh_curve,
                    "position_curve": curves_empty,
                    "realized_pnl_curve": curves_empty,
                    "unrealized_pnl_curve": curves_empty,
                    "total_pnl_curve": curves_empty,
                    "sharpe_ratio": float(sharpe_ratio),
                    "final_pnl": float(final_total),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "total_commission": float(total_commission),
                    "n_trades": int(n_trades),
                    "max_drawdown": float(max_drawdown),
                }
            return results

        if stride != 1:
            raise ValueError(
                "Python fallback run() поддерживает только stride==1 (используйте use_numba=True для stride>1)"
            )

        for i in range(n_all):
            low = float(low_np[i])
            high = float(high_np[i])
            close = float(close_np[i])

            self.background_monitoring(low, high, close)
            self.cancel_old_orders()
            self.place_new_orders(close, float(mid_np[i]), float(spread_np[i]))

        pnl_series = pd.Series(self.total_pnl_curve)
        returns = pnl_series.diff().dropna() / (position_size + 1e-12)
        periods_per_year = (365 * 24 * 60 * 60) / max(bar_seconds, 0.1)

        sharpe_ratio = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

        cum = np.array(self.total_pnl_curve, dtype=float)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

        last_close = float(close_np[-1])
        gross_pnl = (
            self.realized_pnl
            + (self.position * (last_close - self.avg_entry_price) if self.position != 0 else 0.0)
        )
        net_pnl = gross_pnl - self.total_commission
        final_total = self.total_pnl_curve[-1] if self.total_pnl_curve else 0.0

        return {
            "bh_curve": bh_curve,
            "position_curve": self.position_curve,
            "realized_pnl_curve": self.realized_pnl_curve,
            "unrealized_pnl_curve": self.unrealized_pnl_curve,
            "total_pnl_curve": self.total_pnl_curve,
            "sharpe_ratio": sharpe_ratio,
            "final_pnl": final_total,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_commission": self.total_commission,
            "n_trades": self.n_trades,
            "max_drawdown": max_drawdown,
        }


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
    bar_seconds: float = 5.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 0.0,
    stride: int = 1,
    method: str = "bayesian",
    de_maxiter: int = 60,
    de_popsize: int = 12,
    de_workers: int = 1,
    de_seed: int = 42,
) -> Dict[str, Any]:
    """
    Подбирает (alpha,beta,gamma,epsilon), максимизируя Sharpe.
    По умолчанию GP-Bayesian (scikit-optimize); симуляция через NumPy/Numba (без медленных .iloc в цикле).

    method:
      - 'bayesian' — gp_minimize (n_calls, n_initial_points)
      - 'de' — scipy differential_evolution (параллель workers>1 возможен при скромном размере данных)
    stride — прореживание баров только для этого поиска (метрики отличаются от полной выборки).
    """
    if not numba_available():
        raise RuntimeError(
            "optimize_parameters_fast требует numba-симулятор (pip install numba)."
        )

    lows = np.ascontiguousarray(ohlcv["low"].to_numpy(dtype=np.float64, copy=False))
    highs = np.ascontiguousarray(ohlcv["high"].to_numpy(dtype=np.float64, copy=False))
    closes = np.ascontiguousarray(ohlcv["close"].to_numpy(dtype=np.float64, copy=False))
    mids = np.ascontiguousarray(np.asarray(mid_predicts, dtype=np.float64))
    spreads = np.ascontiguousarray(np.asarray(spread_predicts, dtype=np.float64))
    empty_curve = np.empty(0, dtype=np.float64)

    def simulate_once(theta: Tuple[float, float, float, float]):
        a, b, g, e = theta
        sr, *_rest = simulate_mm_backtest(
            lows,
            highs,
            closes,
            mids,
            spreads,
            float(a),
            float(b),
            float(g),
            float(e),
            float(max_position),
            float(position_qty),
            float(commission_bps),
            float(slippage_bps),
            float(position_size),
            float(bar_seconds),
            1.0,
            int(stride),
            False,
            empty_curve,
            empty_curve,
            empty_curve,
            empty_curve,
        )
        return float(sr)

    bayesian_bounds_skopt = [
        alpha_range,
        beta_range,
        gamma_range,
        epsilon_range,
    ]

    base_out = {
        "max_position": max_position,
        "position_qty": position_qty,
        "bar_seconds": bar_seconds,
        "stride": stride,
        "method": method,
    }

    method_l = method.lower().strip()

    if method_l == "bayesian":
        try:
            from skopt.space import Real
            from skopt import gp_minimize
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "method='bayesian' требует scikit-optimize: pip install scikit-optimize"
            ) from e

        space = [
            Real(bayesian_bounds_skopt[0][0], bayesian_bounds_skopt[0][1], name="alpha"),
            Real(bayesian_bounds_skopt[1][0], bayesian_bounds_skopt[1][1], name="beta"),
            Real(bayesian_bounds_skopt[2][0], bayesian_bounds_skopt[2][1], name="gamma"),
            Real(bayesian_bounds_skopt[3][0], bayesian_bounds_skopt[3][1], name="epsilon"),
        ]

        result = gp_minimize(
            lambda xs: -simulate_once((float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]))),
            space,
            n_calls=int(n_calls),
            n_initial_points=int(n_initial_points),
            random_state=42,
            verbose=True,
        )
        xv = tuple(float(v) for v in result.x)
        best_sr = simulate_once(xv)

        base_out.update(
            {
                "alpha": xv[0],
                "beta": xv[1],
                "gamma": xv[2],
                "epsilon": xv[3],
                "best_sharpe": best_sr,
            }
        )
        return base_out

    if method_l == "de":
        try:
            from scipy.optimize import differential_evolution
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "method='de' требует scipy: pip install scipy"
            ) from e

        bounds_scipy = [
            tuple(bayesian_bounds_skopt[k]) for k in range(4)
        ]

        def de_obj(xx: np.ndarray) -> float:
            return float(-simulate_once(tuple(map(float, xx))))

        res = differential_evolution(
            de_obj,
            bounds_scipy,
            maxiter=int(de_maxiter),
            popsize=int(de_popsize),
            polish=True,
            workers=int(de_workers),
            updating="deferred" if int(de_workers) != 1 else "immediate",
            seed=int(de_seed),
            strategy="best1bin",
        )
        xv = tuple(float(v) for v in res.x)
        best_sr = simulate_once(xv)
        base_out.update(
            {
                "alpha": xv[0],
                "beta": xv[1],
                "gamma": xv[2],
                "epsilon": xv[3],
                "best_sharpe": best_sr,
                "de_nit": int(res.nit),
                "de_nfev": int(res.nfev),
                "success": bool(res.success),
                "message": str(res.message),
            }
        )
        return base_out

    raise ValueError("Неизвестный method: используйте 'bayesian' или 'de'.")


def optimize_parameters_pareto(
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
    *,
    population_size: int = 80,
    n_generations: int = 35,
    bar_seconds: float = 5.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 0.0,
    stride: int = 1,
    rng_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    NSGA-II: Парето-фронт по двум целям (максимизация Sharpe И финальный total_pnl после прогона).
    Внутри задачи переводится в минимизацию отрицаний двух показателей.
    Требует numba-симулятор (см. mm_backtest_numba).
    """
    if not numba_available():
        raise RuntimeError("optimize_parameters_pareto требует numba-симулятор.")

    lows = np.ascontiguousarray(ohlcv["low"].to_numpy(dtype=np.float64, copy=False))
    highs = np.ascontiguousarray(ohlcv["high"].to_numpy(dtype=np.float64, copy=False))
    closes = np.ascontiguousarray(ohlcv["close"].to_numpy(dtype=np.float64, copy=False))
    mids = np.ascontiguousarray(np.asarray(mid_predicts, dtype=np.float64))
    spreads = np.ascontiguousarray(np.asarray(spread_predicts, dtype=np.float64))
    empty_curve = np.empty(0, dtype=np.float64)

    def eval_row(xx: np.ndarray) -> Tuple[float, float]:
        row = xx.astype(np.float64, copy=False)
        sr, fp, _gp, _net, _mdd, _ntr, _tcomm, _nsteps = simulate_mm_backtest(
            lows,
            highs,
            closes,
            mids,
            spreads,
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(max_position),
            float(position_qty),
            float(commission_bps),
            float(slippage_bps),
            float(position_size),
            float(bar_seconds),
            1.0,
            int(stride),
            False,
            empty_curve,
            empty_curve,
            empty_curve,
            empty_curve,
        )

        sharpe_goal = sr
        pnl_goal = fp
        return float(sharpe_goal), float(pnl_goal)

    def evaluate_population(pop_arr: np.ndarray) -> np.ndarray:
        scores = np.empty((pop_arr.shape[0], 2), dtype=np.float64)
        for i in range(pop_arr.shape[0]):
            s, fp = eval_row(pop_arr[i])
            scores[i, 0] = -s
            scores[i, 1] = -fp
        return scores

    xl = np.asarray(
        [alpha_range[0], beta_range[0], gamma_range[0], epsilon_range[0]],
        dtype=np.float64,
    )
    xu = np.asarray(
        [alpha_range[1], beta_range[1], gamma_range[1], epsilon_range[1]],
        dtype=np.float64,
    )

    res = run_nsga2(
        evaluate_population,
        xl,
        xu,
        int(population_size),
        int(n_generations),
        seed=rng_seed,
    )

    solutions: List[Dict[str, float]] = []
    for row, fv in zip(res.pareto_params, res.pareto_F):
        xv = tuple(float(z) for z in row)
        solutions.append(
            {
                "alpha": xv[0],
                "beta": xv[1],
                "gamma": xv[2],
                "epsilon": xv[3],
                "sharpe_ratio": float(-fv[0]),
                "final_pnl": float(-fv[1]),
                "max_position": float(max_position),
                "position_qty": float(position_qty),
            }
        )

    return {
        "pareto_front": solutions,
        "population_size": int(population_size),
        "n_generations": int(n_generations),
        "stride": stride,
        "bar_seconds": bar_seconds,
        "bounds": {
            "alpha_range": tuple(map(float, alpha_range)),
            "beta_range": tuple(map(float, beta_range)),
            "gamma_range": tuple(map(float, gamma_range)),
            "epsilon_range": tuple(map(float, epsilon_range)),
        },
    }


