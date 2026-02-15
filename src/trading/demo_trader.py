import asyncio
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter
from src.trading.backtest_trader import BaseTrader
from src.parser.live_collector import LiveCollector
from src.mlcore.dataloader import get_processed_data, calculate_features



class DemoTrader(BaseTrader):
    def __init__(
        self,
        mid_model: XGBRegressor,
        spread_model: XGBRegressor,
        alpha: float,
        beta: float,
        gamma: float,
        epsilon: float,
        max_position: float = 0.01,
        position_qty: float = 0.002,
        log_dir: str = "./runs/demo",
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            max_position=max_position,
            position_qty=position_qty,
        )

        self.parser = None
        self.mid_model = mid_model
        self.spread_model = spread_model
        
        self.step = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.background_task = None
        self.bh_reference_price = None

    def get_current_price(self) -> float:
        df = self.parser.get_dataframe("aggTrades")
        if df.empty:
            raise RuntimeError("No aggTrades data available")
        return df.iloc[-1]["price"]
    
    async def background_monitoring(self) -> None:
        print("✅ Warmup finished. Now starting background monitoring...")
        while True:
            try:
                current_price = self.get_current_price()

                if self.bh_reference_price is None:
                    self.bh_reference_price = current_price
                    print(f"📌 Buy&Hold reference price зафиксирована: {self.bh_reference_price}")

                # 1. Проверяем исполнение ордеров
                if self.buy_order and current_price <= self.buy_order['price']:
                    order_price = self.buy_order['price']
                    self.handle_buy_order()
                    print(f"✅ Buy order filled at: {order_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")
                if self.sell_order and current_price >= self.sell_order['price']:
                    order_price = self.sell_order['price']
                    self.handle_sell_order()
                    print(f"✅ Sell order filled at: {order_price}. Position: {self.position}, Realized PnL: {self.realized_pnl}")

                # Buy&Hold в абсолюте и в относительном виде
                if self.bh_reference_price is not None and self.bh_reference_price > 0:
                    bh_absolute = (current_price / self.bh_reference_price - 1) * 100  # в процентах
                else:
                    bh_absolute = 0.0

                # 2. Логируем в TensorBoard
                self.writer.add_scalar("BuyHold_PnL_%", bh_absolute, self.step)
                self.writer.add_scalar("Realized_PnL", self.realized_pnl, self.step)
                self.writer.add_scalar("Position", self.position, self.step)
                self.step += 1
            except Exception as e:
                print(f"⚠️ Background monitoring error: {e}")

            await asyncio.sleep(0.05)

    def cancel_old_orders(self):
        if self.buy_order:
            print(f"❌ Cancelled old BUY order at: {self.buy_order['price']}")
            self.buy_order = None
        if self.sell_order:
            print(f"❌ Cancelled old SELL order at: {self.sell_order['price']}")
            self.sell_order = None

    def place_new_orders(self, last_price: float, mid_pred: float, spread_pred: float) -> None:
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
            print(f"📈 Placed BUY order at: {buy_price}")
        
        if can_sell:
            self.sell_order = {
                'price': sell_price,
                'origQty': self.position_qty,
            }
            print(f"📉 Placed SELL order at: {sell_price}")

    async def run(self, symbol: str = "BTCUSDC") -> None:
        # Запускаем парсер данных в реальном времени
        self.parser = LiveCollector(symbol=symbol, retention_seconds=300)
        parser_task = asyncio.create_task(self.parser.start())

        try:
            # Ждём пока соберётся достаточное количество данных
            print("⏳ Warming up data collector for 300 seconds...")
            pbar = tqdm(total=300, desc="Collecting data...", unit="s")
            for _ in range(300):
                await asyncio.sleep(1)
                pbar.update(1)
            pbar.close()

            self.background_task = asyncio.create_task(self.background_monitoring())

            print("🟢 Trading started!")
            df = get_processed_data(source='live', live_collector=self.parser)
            last_bar_time = df.iloc[-1]['exchange_ts']
            while True:
                try:
                    # Получаем данные всегда 
                    df = get_processed_data(source='live', live_collector=self.parser)
                    current_bar_time = df.iloc[-1]['exchange_ts']

                    # Как только получили новую временную метку
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
                        self.cancel_old_orders()

                        # Выставляем новые ордеры
                        self.place_new_orders(last_price, mid_pred, spread_pred)

                        # Запоминаем текущую временную метку
                        last_bar_time = df.iloc[-1]['exchange_ts']
                        
                except Exception as e:
                    print(f"⚠️ Trading loop error: {e}")

                await asyncio.sleep(0.05)

        finally:
            print("🛑 Shutting down DemoTraider...")
            if self.background_task and not self.background_task.done():
                self.background_task.cancel()
                try:
                    await self.background_task
                except asyncio.CancelledError:
                    pass

            await self.parser.stop()
            if not parser_task.done():
                parser_task.cancel()
                try:
                    await parser_task
                except asyncio.CancelledError:
                    pass
            self.writer.close()