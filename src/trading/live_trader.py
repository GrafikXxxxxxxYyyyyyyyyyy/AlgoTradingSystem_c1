import asyncio
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from xgboost import XGBRegressor
from binance import Client, AsyncClient
from torch.utils.tensorboard import SummaryWriter
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


    async def background_monitoring(self) -> None:
        print("✅ Warmup finished. Now starting background monitoring...")

        while not self.shutdown:
            async with self.state_lock:
                try:
                    # 1. Проверяем исполнение ордеров
                    orders = await self.client.futures_get_open_orders(symbol=self.symbol.upper())
                    order_ids = {str(o['orderId']) for o in orders}

                    buy_done = self.buy_order and str(self.buy_order['orderId']) not in order_ids
                    sell_done = self.sell_order and str(self.sell_order['orderId']) not in order_ids
                    
                    if buy_done:
                        order_price = float(self.buy_order['price'])
                        self.handle_buy_order()
                        print(f"✅ Buy order filled at: {order_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")

                    if sell_done:
                        order_price = float(self.sell_order['price'])
                        self.handle_sell_order()
                        print(f"✅ Sell order filled at: {order_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")

                    # 2. Логируем в TensorBoard
                    self.writer.add_scalar("Realized_PnL", self.realized_pnl, self.step)
                    self.writer.add_scalar("Position", self.position, self.step)
                    self.step += 1
                except Exception as e:
                    print(f"⚠️ Background monitoring error: {e}")

            await asyncio.sleep(0.05)

        
    async def cancel_old_orders(self) -> None:
        async with self.state_lock:
            if self.buy_order:
                order_id = self.buy_order['orderId']
                try:
                    await self.client.futures_cancel_order(
                        symbol=self.symbol.upper(), 
                        orderId=order_id
                    )
                    self.buy_order = None
                    print(f"❌ Ордер на покупку id={order_id} отменён")
                except Exception as e:
                    if "Order does not exist" not in str(e):
                        print(f"❗ Ошибка отмены id={order_id}: {e}")

            if self.sell_order:
                order_id = self.sell_order['orderId']
                try:
                    await self.client.futures_cancel_order(
                        symbol=self.symbol.upper(), 
                        orderId=order_id
                    )
                    self.sell_order = None
                    print(f"❌ Ордер на продажу id={order_id} отменён")
                except Exception as e:
                    if "Order does not exist" not in str(e):
                        print(f"❗ Ошибка отмены id={order_id}: {e}")


    async def place_new_orders(self, last_price: float, mid_pred: float, spread_pred: float) -> None:
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

            if can_buy:
                try: 
                    self.buy_order = await self.client.futures_create_order(
                        symbol=self.symbol.upper(),
                        type='LIMIT',
                        timeInForce='GTX',
                        side="BUY",
                        quantity=self.position_qty,
                        price=buy_price,
                        postOnly=True,
                        reduceOnly=False,
                    )
                    print(f"🆕 Выставлен ордер на покупку {self.position_qty} @ {buy_price} (ID: {self.buy_order['orderId']})")
                except Exception as e:
                    print(f"❗ Ошибка выставления ордера на покупку: {e}")

            if can_sell:
                try: 
                    self.sell_order = await self.client.futures_create_order(
                        symbol=self.symbol.upper(),
                        type='LIMIT',
                        timeInForce='GTX',
                        side="SELL",
                        quantity=self.position_qty,
                        price=sell_price,
                        postOnly=True,
                        reduceOnly=False,
                    )
                    print(f"🆕 Выставлен ордер на продажу {self.position_qty} @ {sell_price} (ID: {self.sell_order['orderId']})")
                except Exception as e:
                    print(f"❗ Ошибка выставления ордера на продажу: {e}")


    async def run(self, symbol: str = "BTCUSDC") -> None:
        # 1. Сохраняем символ с которым работаем
        self.symbol = symbol

        # 2. Устанавливаем торговое плечо
        try:
            await self.client.futures_change_leverage(symbol=symbol.upper(), leverage=self.leverage)
            print(f"⚖️ Плечо для {symbol} установлено: x{self.leverage}")
        except Exception as e:
            print(f"❗ Ошибка установки плеча: {e}")
            raise

        # 3. Инициализируем и запускаем live-парсер данных
        self.parser = LiveCollector(symbol=symbol.upper(), retention_seconds=300)
        parser_task = asyncio.create_task(self.parser.start())

        try:
            # 4. Ждём пока соберётся достаточное количество данных 
            print("⏳ Warming up data collector for 300 seconds...")
            pbar = tqdm(total=300, desc="Collecting data...", unit="s")
            for _ in range(300):
                await asyncio.sleep(1)
                pbar.update(1)
            pbar.close()

            # 5. Создаём фоновые задачи для управления ордерами и позициями
            self.background_task = asyncio.create_task(self.background_monitoring())

            # 6. Запускаем торговый цикл
            print("🟢 Trading started!")
            df = get_processed_data(source='live', live_collector=self.parser)
            last_bar_time = df.iloc[-1]['exchange_ts']
            while not self.shutdown:
                try:
                    df = get_processed_data(source='live', live_collector=self.parser)
                    current_bar_time = df.iloc[-1]['exchange_ts']

                    if current_bar_time > last_bar_time:
                        # Получаем предпоследнюю цену
                        last_price = df.iloc[-2]['close']
                        # Генерируем признаки
                        features = calculate_features(df, filename="all_features")
                        # Получаем признаки за ПРЕДПОСЛЕДНЮЮ метку
                        model_input = features.iloc[-2:-1].copy()
                        # Делаем прогноз моделeй
                        mid_pred = self.mid_model.predict(model_input)[0]
                        spread_pred = self.spread_model.predict(model_input)[0]

                        # Отменяем старые ордеры
                        await self.cancel_old_orders()

                        # Выставляем новые ордеры
                        await self.place_new_orders(last_price, mid_pred, spread_pred)

                        last_bar_time = df.iloc[-1]['exchange_ts']

                except Exception as e:
                    print(f"⚠️ Trading loop error: {e}")

                await asyncio.sleep(0.05)

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