"""
Fast market-making backtest core (Numba). Mirrors BacktestTrader bar logic without pandas .iloc overhead.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def _optimal_quotes(
        last_price: float,
        mid_pred: float,
        spread_pred: float,
        position: float,
        avg_entry_price: float,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: float,
        max_position: float,
        min_spread_bps: float,
    ):
        optimal_spread = alpha * spread_pred * last_price
        directional_drift = beta * last_price * (np.exp(mid_pred) - 1.0)
        inv_mp = 1.0 / max_position
        positional_skew = gamma * position * inv_mp * spread_pred * last_price

        half_spread = optimal_spread
        ms = last_price * (min_spread_bps * 1e-4)
        if half_spread < ms:
            half_spread = ms

        bid_adjustment = 0.0
        ask_adjustment = 0.0
        if avg_entry_price != 0.0 and position != 0.0:
            if position > 0.0:
                ask_adjustment = epsilon * (last_price - avg_entry_price)
            else:
                bid_adjustment = epsilon * (last_price - avg_entry_price)

        bid_price = last_price - half_spread + directional_drift - positional_skew + bid_adjustment
        ask_price = last_price + half_spread + directional_drift - positional_skew + ask_adjustment

        bid_price = min(last_price - 0.1, bid_price)
        ask_price = max(last_price + 0.1, ask_price)
        if ask_price <= bid_price:
            ask_price = bid_price + max(0.1, last_price * (min_spread_bps * 1e-4))
        return bid_price, ask_price

    @njit(cache=True)
    def _fill_px_commission(order_price: float, qty: float, side_buy: bool, commission_bps: float, slippage_bps: float):
        bps = 1e-4
        if side_buy:
            fill_price = order_price * (1.0 + slippage_bps * bps)
        else:
            fill_price = order_price * (1.0 - slippage_bps * bps)
        commission = qty * fill_price * (commission_bps * bps)
        return fill_price, commission

    @njit(cache=True)
    def simulate_mm_backtest(
        low,
        high,
        close,
        mid_pred,
        spread_pred,
        alpha,
        beta,
        gamma,
        epsilon,
        max_position,
        position_qty,
        commission_bps,
        slippage_bps,
        position_size,
        bar_seconds,
        min_spread_bps,
        stride: int,
        store_curves: bool,
        position_curve,
        realized_curve,
        unrealized_curve,
        total_pnl_curve,
    ):
        n_all = len(low)

        realized_pnl = 0.0
        position = 0.0
        avg_entry = 0.0
        total_commission = 0.0
        n_trades = 0

        buy_px = np.nan
        buy_active = False
        sell_px = np.nan
        sell_active = False

        sharpe_mean = 0.0
        sharpe_m2 = 0.0
        sharpe_cnt = 0
        prev_total = 0.0
        final_total_tp = 0.0

        run_max_total = -1.0e30
        max_dd = 0.0

        step_idx = 0
        last_close = close[0]

        for i_raw in range(n_all):
            if stride > 1 and (i_raw % stride) != 0:
                continue

            low_i = low[i_raw]
            high_i = high[i_raw]
            close_i = close[i_raw]

            mid_i = mid_pred[i_raw]
            sp_i = spread_pred[i_raw]

            if buy_active and low_i <= buy_px:
                qty = position_qty
                fill_price, comm = _fill_px_commission(buy_px, qty, True, commission_bps, slippage_bps)
                total_commission += comm
                n_trades += 1

                if position >= 0.0:
                    new_position = position + qty
                    if new_position != 0.0:
                        avg_entry = (position * avg_entry + qty * fill_price) / new_position
                    else:
                        avg_entry = fill_price
                    position = new_position
                else:
                    qty_to_close = min(-position, qty)
                    realized_pnl += qty_to_close * (avg_entry - fill_price)
                    position += qty_to_close

                    remaining_qty = qty - qty_to_close
                    if remaining_qty > 0.0:
                        position = remaining_qty
                        avg_entry = fill_price
                    elif position == 0.0:
                        avg_entry = 0.0

                realized_pnl -= comm
                buy_active = False

            if sell_active and high_i >= sell_px:
                qty = position_qty
                fill_price, comm = _fill_px_commission(sell_px, qty, False, commission_bps, slippage_bps)
                total_commission += comm
                n_trades += 1

                if position <= 0.0:
                    new_position = position - qty
                    if new_position != 0.0:
                        avg_entry = (position * avg_entry + (-qty) * fill_price) / new_position
                    else:
                        avg_entry = fill_price
                    position = new_position
                else:
                    qty_to_close = min(position, qty)
                    realized_pnl += qty_to_close * (fill_price - avg_entry)
                    position -= qty_to_close

                    remaining_qty = qty - qty_to_close
                    if remaining_qty > 0.0:
                        position = -remaining_qty
                        avg_entry = fill_price
                    elif position == 0.0:
                        avg_entry = 0.0

                realized_pnl -= comm
                sell_active = False

            if position != 0.0:
                unrealized = position * (close_i - avg_entry)
            else:
                unrealized = 0.0

            total_pnl = realized_pnl + unrealized

            if store_curves and step_idx < len(total_pnl_curve):
                position_curve[step_idx] = position
                realized_curve[step_idx] = realized_pnl
                unrealized_curve[step_idx] = unrealized
                total_pnl_curve[step_idx] = total_pnl

            scaled_ret = (total_pnl - prev_total) / (position_size + 1e-12)
            if step_idx > 0:
                sharpe_cnt += 1
                d = scaled_ret - sharpe_mean
                sharpe_mean += d / sharpe_cnt
                d2 = scaled_ret - sharpe_mean
                sharpe_m2 += d * d2

            prev_total = total_pnl

            if total_pnl > run_max_total:
                run_max_total = total_pnl
            dd = run_max_total - total_pnl
            if dd > max_dd:
                max_dd = dd

            bp, ap = _optimal_quotes(
                close_i, mid_i, sp_i, position, avg_entry, alpha, beta, gamma, epsilon, max_position, min_spread_bps
            )

            buy_active = False
            sell_active = False
            if position + position_qty <= max_position:
                buy_px = bp
                buy_active = True
            if position - position_qty >= -max_position:
                sell_px = ap
                sell_active = True

            last_close = close_i
            step_idx += 1
            final_total_tp = total_pnl

        sharpe_ratio = 0.0
        if sharpe_cnt > 1:
            var_sample = sharpe_m2 / (sharpe_cnt - 1.0)
            if var_sample > 0.0:
                std = np.sqrt(var_sample)
                periods_per_year = (365.0 * 24.0 * 60.0 * 60.0) / max(bar_seconds * stride, 0.1)
                sharpe_ratio = (sharpe_mean / std) * np.sqrt(periods_per_year)

        if position != 0.0:
            gross_pnl = realized_pnl + position * (last_close - avg_entry)
        else:
            gross_pnl = realized_pnl

        net_pnl = gross_pnl - total_commission
        final_total = final_total_tp

        return sharpe_ratio, final_total, gross_pnl, net_pnl, max_dd, n_trades, total_commission, step_idx


else:

    def simulate_mm_backtest(*args, **kwargs):
        raise RuntimeError("numba is required for simulate_mm_backtest")
