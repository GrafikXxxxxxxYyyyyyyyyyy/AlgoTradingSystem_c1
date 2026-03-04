import asyncio
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from tqdm import tqdm
from pathlib import Path
from xgboost import XGBRegressor
from binance import AsyncClient
from tensorboardX import SummaryWriter
from src.trading.backtest_trader import BaseTrader
from src.parser.live_collector import LiveCollector
from src.mlcore.dataloader import get_processed_data, calculate_features



class LiveTrader(BaseTrader):
    def __init__(
        self,
        mid_model: XGBRegressor,
        spread_model: XGBRegressor,
        api_key: str,
        api_secret: str,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: float,
        leverage: int = 10,
        max_position: float = 0.01,
        position_qty: float = 0.002,
        log_dir: str = "./runs/live",
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            max_position=max_position,
            position_qty=position_qty,
        )

        self.client = AsyncClient(
            api_key=api_key,
            api_secret=api_secret,
        )
        self.leverage = leverage

        self.parser = None
        self.mid_model = mid_model
        self.spread_model = spread_model

        self.step = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.background_task = None
        self.bh_reference_price = None

        self.shutdown = False
        self.state_lock = asyncio.Lock()


    async def sync_position_from_exchange(self) -> None:
        """Синхронизирует position и avg_entry_price с биржей (полезно при старте после перезапуска)."""
        try:
            acc = await self.client.futures_account()
            for pos in acc.get("positions", []):
                if pos.get("symbol") != self.symbol.upper():
                    continue
                amt = float(pos.get("positionAmt", 0) or 0)
                entry = float(pos.get("entryPrice", 0) or 0)
                if amt != 0:
                    self.position = amt
                    self.avg_entry_price = entry
                    print(f"📌 Позиция с биржи: {self.position}, avg entry: {self.avg_entry_price}")
                break
        except Exception as e:
            print(f"⚠️ Не удалось синхронизировать позицию с биржей: {e}")

    async def _get_order_fill_info(self, order_id: int) -> Tuple[float, float]:
        """Запрашивает у биржи фактически исполненный объём и среднюю цену исполнения. Возвращает (executed_qty, avg_price)."""
        try:
            order = await self.client.futures_get_order(symbol=self.symbol.upper(), orderId=order_id)
            executed = float(order.get("executedQty", 0) or 0)
            avg_price = float(order.get("avgPrice", 0) or 0)
            if avg_price <= 0 and executed > 0:
                avg_price = float(order.get("price", 0) or 0)
            return executed, avg_price
        except Exception as e:
            print(f"⚠️ Не удалось получить детали ордера {order_id}: {e}")
            return 0.0, 0.0

    async def background_monitoring(self) -> None:
        print("✅ Warmup finished. Now starting background monitoring...")

        while not self.shutdown:
            async with self.state_lock:
                try:
                    orders = await self.client.futures_get_open_orders(symbol=self.symbol.upper())
                    order_ids = {str(o["orderId"]) for o in orders}

                    buy_done = self.buy_order and str(self.buy_order["orderId"]) not in order_ids
                    sell_done = self.sell_order and str(self.sell_order["orderId"]) not in order_ids

                    if buy_done:
                        order_id = self.buy_order["orderId"]
                        executed_qty, avg_price = await self._get_order_fill_info(order_id)
                        if executed_qty > 0:
                            self.handle_buy_order(
                                fill_price=avg_price if avg_price > 0 else float(self.buy_order["price"]),
                                commission=0.0,
                                actual_filled_qty=executed_qty,
                            )
                            print(f"✅ Buy filled: qty={executed_qty}, avg={avg_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")
                        else:
                            self.buy_order = None

                    if sell_done:
                        order_id = self.sell_order["orderId"]
                        executed_qty, avg_price = await self._get_order_fill_info(order_id)
                        if executed_qty > 0:
                            self.handle_sell_order(
                                fill_price=avg_price if avg_price > 0 else float(self.sell_order["price"]),
                                commission=0.0,
                                actual_filled_qty=executed_qty,
                            )
                            print(f"✅ Sell filled: qty={executed_qty}, avg={avg_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")
                        else:
                            self.sell_order = None

                    self.writer.add_scalar("Realized_PnL", self.realized_pnl, self.step)
                    self.writer.add_scalar("Position", self.position, self.step)
                    self.step += 1
                except Exception as e:
                    print(f"⚠️ Background monitoring error: {e}")

            await asyncio.sleep(0.05)

        
    async def cancel_old_orders(self) -> None:
        """Отмена всех ордеров по символу одним запросом (один round-trip). Новые не выставляются до завершения отмены."""
        async with self.state_lock:
            has_orders = self.buy_order is not None or self.sell_order is not None
            buy_id = self.buy_order["orderId"] if self.buy_order else None
            sell_id = self.sell_order["orderId"] if self.sell_order else None
            self.buy_order = None
            self.sell_order = None
        if not has_orders:
            return
        sym = self.symbol.upper()
        try:
            await self.client.futures_cancel_all_open_orders(symbol=sym)
        except Exception as e:
            if "Order does not exist" not in str(e) and "Unknown order" not in str(e):
                print(f"❗ Ошибка отмены ордеров: {e}")
        else:
            if buy_id or sell_id:
                print(f"❌ Ордера отменены (buy={buy_id}, sell={sell_id})")


    async def place_new_orders(self, last_price: float, mid_pred: float, spread_pred: float) -> None:
        """Выставление buy и sell параллельно; lock не держится во время запросов к API."""
        async with self.state_lock:
            buy_price, sell_price = self.calculate_optimal_quotes(
                last_price=last_price,
                mid_pred=mid_pred,
                spread_pred=spread_pred,
            )
            buy_price = round(buy_price, 1)
            sell_price = round(sell_price, 1)
            can_buy = self.position + self.position_qty <= self.max_position
            can_sell = self.position - self.position_qty >= -self.max_position
            sym = self.symbol.upper()
            qty = self.position_qty

        async def place_buy():
            if not can_buy:
                return None
            try:
                return await self.client.futures_create_order(
                    symbol=sym,
                    type="LIMIT",
                    timeInForce="GTX",
                    side="BUY",
                    quantity=qty,
                    price=buy_price,
                    postOnly=True,
                    reduceOnly=False,
                )
            except Exception as e:
                print(f"❗ Ошибка выставления ордера на покупку: {e}")
                return None

        async def place_sell():
            if not can_sell:
                return None
            try:
                return await self.client.futures_create_order(
                    symbol=sym,
                    type="LIMIT",
                    timeInForce="GTX",
                    side="SELL",
                    quantity=qty,
                    price=sell_price,
                    postOnly=True,
                    reduceOnly=False,
                )
            except Exception as e:
                print(f"❗ Ошибка выставления ордера на продажу: {e}")
                return None

        buy_res, sell_res = await asyncio.gather(place_buy(), place_sell())
        async with self.state_lock:
            if buy_res is not None:
                self.buy_order = buy_res
            if sell_res is not None:
                self.sell_order = sell_res
        if buy_res is not None:
            print(f"🆕 Выставлен ордер на покупку {qty} @ {buy_price} (ID: {buy_res['orderId']})")
        if sell_res is not None:
            print(f"🆕 Выставлен ордер на продажу {qty} @ {sell_price} (ID: {sell_res['orderId']})")


    async def run(
        self,
        symbol: str = "BTCUSDC",
        grid_resolution_ms: Optional[int] = None,
        quote_interval_sec: float = 0.05,
    ) -> None:
        """
        grid_resolution_ms: интервал сетки в мс (5000 по умолчанию; 100 для высокой частоты).
        quote_interval_sec: пауза в цикле (0.05 = 50 ms).
        """
        resolution = grid_resolution_ms if grid_resolution_ms is not None else 5000
        self.symbol = symbol

        try:
            await self.client.futures_change_leverage(symbol=symbol.upper(), leverage=self.leverage)
            print(f"⚖️ Плечо для {symbol} установлено: x{self.leverage}")
        except Exception as e:
            print(f"❗ Ошибка установки плеча: {e}")
            raise

        await self.sync_position_from_exchange()

        self.parser = LiveCollector(symbol=symbol.upper(), retention_seconds=300)
        parser_task = asyncio.create_task(self.parser.start())

        try:
            print("⏳ Warming up data collector for 300 seconds...")
            pbar = tqdm(total=300, desc="Collecting data...", unit="s")
            for _ in range(300):
                await asyncio.sleep(1)
                pbar.update(1)
            pbar.close()

            self.background_task = asyncio.create_task(self.background_monitoring())
            print("🟢 Trading started!")
            df = get_processed_data(
                source="live",
                live_collector=self.parser,
                grid_resolution_ms=resolution,
            )
            last_bar_time = df.iloc[-1]["exchange_ts"] if not df.empty else None

            while not self.shutdown:
                try:
                    df = get_processed_data(
                        source="live",
                        live_collector=self.parser,
                        grid_resolution_ms=resolution,
                    )
                    if df.empty or len(df) < 2:
                        await asyncio.sleep(quote_interval_sec)
                        continue
                    current_bar_time = df.iloc[-1]["exchange_ts"]
                    # Отменяем и выставляем новые ордера только при появлении нового бара
                    if last_bar_time is not None and current_bar_time > last_bar_time:
                        cancel_task = asyncio.create_task(self.cancel_old_orders())
                        # Текущий бар (iloc[-1]) ещё не закрыт — используем только закрытый бар iloc[-2] для фичей и close
                        last_price = float(df.iloc[-2]["close"])
                        features = calculate_features(df, filename="all_features")
                        model_input = features.iloc[-2:-1].copy()
                        if model_input.isna().any().any():
                            model_input = model_input.fillna(0)
                        mid_pred = float(self.mid_model.predict(model_input)[0])
                        spread_pred = float(self.spread_model.predict(model_input)[0])
                        await cancel_task
                        await self.place_new_orders(last_price, mid_pred, spread_pred)
                        last_bar_time = current_bar_time
                    await asyncio.sleep(quote_interval_sec)
                except Exception as e:
                    print(f"⚠️ Trading loop error: {e}")
                await asyncio.sleep(quote_interval_sec)

        finally:
            print("🛑 Завершение...")
            self.shutdown = True

            # Отменяем все ордера
            try:
                open_orders = await self.client.futures_get_open_orders(symbol=self.symbol.upper())
                for order in open_orders:
                    await self.client.futures_cancel_order(symbol=self.symbol.upper(), orderId=order['orderId'])
            except:
                pass

            # Закрываем открытые позиции рыночным ордером (осторожно!)
            if self.position != 0:
                qty = abs(self.position)
                side = 'SELL' if self.position > 0 else 'BUY'
                try:
                    await self.client.futures_create_order(
                        symbol=self.symbol.upper(),
                        type='MARKET',
                        side=side,
                        quantity=qty,
                        reduceOnly=True
                    )
                    print("CloseOperation: позиция закрыта при завершении")
                except Exception as e:
                    print(f"Не удалось закрыть позицию: {e}")

            # Останавливаем фоновые задачи
            if self.background_task and not self.background_task.done():
                self.background_task.cancel()
                try:
                    await self.background_task
                except asyncio.CancelledError:
                    pass
            
            # Останавливаем парсер
            await self.parser.stop()
            if not parser_task.done():
                parser_task.cancel()
                try:
                    await parser_task
                except asyncio.CancelledError:
                    pass
            
            # Закрываем соединение и логирование
            await self.client.close_connection()
            self.writer.close()

            print("✅ LiveTrader остановлен")