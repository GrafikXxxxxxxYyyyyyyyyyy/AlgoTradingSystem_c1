# Отчёт аудита кода: AlgoTradingSystem_c1

## Обзор проекта

Система алгоритмической торговли на Binance Futures: сбор стакана и тиков, фичи, модели (XGBoost для mid/spread), бэктест и live/demo трейдинг.

---

## 1. Критические ошибки (проект не запускается)

### 1.1 Неверный импорт в `collector.py`

**Файл:** `src/parser/collector.py`, строка 114

```python
from storage import WALLogger
```

Должно быть относительный импорт внутри пакета `parser`:

```python
from .storage import WALLogger
```

Иначе при вызове `_recover_from_wal()` будет поиск глобального модуля `storage`, а не `src.parser.storage`.

---

### 1.4 Несовпадение сигнатур процессоров и их вызова

**Файл:** `src/mlcore/processors.py`

- `BaseProcessor.__call__(self, stream_df, need_prev_hour=False)` — **не принимает** `canonical_grid` и **не принимает** `grid_config`.
- Конструкторы процессоров: `AggTradesProcessor()`, `RawTradesProcessor()`, `OrderbookProcessor()` — без аргументов.

**Файл:** `src/mlcore/dataloader.py`

- Строки 76, 204: вызов `processor(df, need_prev_hour=..., canonical_grid=canonical_grid)` — лишний аргумент `canonical_grid` → **TypeError**.
- Строки 104–106: создание процессоров с `grid_config=grid_config`:
  - `AggTradesProcessor(grid_config=grid_config)` и т.д. — конструкторы таких аргументов **не принимают** → **TypeError**.

**Следствие:** при `source="raw"` или `source="live"` в `get_processed_data()` будет исключение.

---

### 1.5 Отсутствующая зависимость `scikit-optimize`

**Файл:** `src/trading/backtest_trader.py`

- Импорт: `from skopt.space import Real`, `from skopt import gp_minimize`
- В `requirements.txt` пакет **scikit-optimize** не указан.

**Следствие:** вызов `optimize_parameters_fast()` приведёт к `ModuleNotFoundError: No module named 'skopt'`.

---

## 2. Архитектурные и логические недостатки

### 2.1 Базовый трейдер: абстрактные методы и реализация

**Файл:** `src/trading/backtest_trader.py`

- В `BaseTrader` объявлены абстрактные методы:
  - `async def background_monitoring(self)`
  - `async def cancel_old_orders(self)`
  - `async def place_new_orders(self, ...)`
- В `BacktestTrader` реализованы **синхронные** версии с другой сигнатурой:
  - `def background_monitoring(self, low, high, close)` — не `async`, добавлены аргументы.
  - `def cancel_old_orders(self)`, `def place_new_orders(self, ...)` — без `async`.

В Python абстрактный метод должен быть переопределён с той же сигнатурой. Здесь переопределение формально не соответствует контракту (другие имена/типы аргументов, синхронность), что затрудняет единообразное использование трейдеров через базовый класс.

---

### 2.2 DemoTrader: синхронные методы вместо async

**Файл:** `src/trading/demo_trader.py`

- `cancel_old_orders` и `place_new_orders` объявлены как обычные функции (`def`), тогда как в `BaseTrader` — `async def`.
- В коде они вызываются без `await` (например, `self.cancel_old_orders()`), поэтому код работает, но контракт с базовым классом нарушен и при полиморфном вызове через `BaseTrader` возможны ошибки.

---

### 2.3 LiveTrader: отсутствующий атрибут `symbol`

**Файл:** `src/trading/live_trader.py`

- В `__init__` не задаётся `self.symbol`.
- `self.symbol` впервые присваивается в `run()`: `self.symbol = symbol`.
- В `sync_position_from_exchange()` и других методах используется `self.symbol.upper()`; до первого вызова `run()` обращение к `self.symbol` вызовет **AttributeError**.

---

### 2.4 Восстановление из WAL: дублирование записей

**Файл:** `src/parser/collector.py`, метод `_recover_from_wal`

- Цикл: `for stream_type in self.storage._schemas.keys():` и внутри — `recovered_count += len(records)`.
- `recovered_count` считает сумму по всем потокам, но логирование говорит «Recovered N records» — по смыслу это общее число восстановленных записей, что корректно.
- После восстановления вызывается `self.storage.flush_all()`. Записи уже попали в буфер через `buffer()`. Важно: при повторном запуске коллектора один и тот же WAL может снова прочитаться и снова попасть в Parquet — нужна политика очистки/ротации WAL после успешного flush, что частично есть в `flush_stream` через `wal_logger.clear()`. Риск — если процесс упадёт после `buffer()`, но до `flush_all()`, при следующем старте записи дублируются.

---

### 2.5 OrderBook: сброс флага синхронизации при snapshot

**Файл:** `src/parser/orderbook.py`

- В `apply_snapshot()` выставляется `self._is_synced = False`, в `apply_diff()` — `True`.
- После применения снапшота стакан ещё не синхронизирован с потоком диффов до прихода первого подходящего диффа. Логика корректна, но имя флага может вводить в заблуждение: «synced» здесь означает «после последнего сообщения мы применили дифф», а не «стакан совпадает с биржей». Для валидации используется отдельная проверка по REST.

---

## 3. Ошибки и опечатки в коде

### 3.1 Опечатка в комментарии

**Файл:** `src/mlcore/processors.py`

- Строка 74: «Отавляем» → «Оставляем».
- Строка 164: то же «Отавляем» → «Оставляем».
- Строка 175: «Пивотим» → «Пивотим» (имелось в виду «пивотим» как pivot — лучше «Делаем pivot»).

---

### 3.3 Опечатка в сообщении

**Файл:** `src/trading/demo_trader.py`, строка 209

- «Shutting down DemoTraider» → «DemoTrader».

---

### 3.4 Дублирование функций в all_features.py / spread_features.py

**Файл:** `src/mlcore/features/all_features.py`

- Функция `intensity_fqi` определена дважды (строки ~136 и ~302).
- Функция `combo6_queue_pressure_sign_times_wli` определена дважды (строки ~361 и ~377).

Вторая реализация перезаписывает первую; если где-то ожидается первая версия по смыслу — возможна путаница. Лучше оставить одну реализацию и удалить дубликат.

---

### 3.5 Потенциальное деление на ноль в evaluation.py

**Файл:** `src/mlcore/evaluation.py`, функция `herding_measure`

```python
herd = np.abs(returns - returns.mean()) / returns.std()
```

Если `returns.std() == 0` (постоянная серия), получится деление на ноль и inf/nan. Стоит добавить проверку или заменить на безопасный знаменатель (например, `returns.std() + 1e-12` или ветвление).

---

### 3.6 Риск KeyError в orderbook_slope_5levels

**Файл:** `src/mlcore/features/all_features.py` (и аналоги в других файлах фичей)

В цикле по `i` проверяется только `f'bid_qty_{i}' in df.columns`; для `ask_qty_{i}'` проверки нет, но затем используется `df[f'ask_qty_{i}']`. Если колонок ask нет — **KeyError**. Нужна проверка и для `ask_qty_{i}`.

---

## 4. Схема данных и типы

### 4.1 Parquet: тип для поля `side`

**Файл:** `src/parser/storage.py`

- Схема для `orderbook_snapshots`: `("side", pa.dictionary(pa.int8(), pa.string()))`.
- В код записываются строки `"bid"` и `"ask"`. PyArrow dictionary поддерживает такие значения, но при первом появлении значения словарь строится автоматически. Если в коде ожидается именно dictionary — нужно убедиться, что при чтении обрабатывается тип словаря (индексы или декодирование).

---

### 4.2 Несогласованность символа по умолчанию

- **Parser config / main:** `CollectorConfig.symbol = "BTCUSDT"`, в `main.py` передаётся символ из аргумента.
- **LiveCollector:** по умолчанию `symbol: str = "BTCUSDC"`.

Разные символы по умолчанию (BTCUSDT vs BTCUSDC) могут привести к путанице и неверному сбору данных при копировании примеров вызова.

---

## 5. Безопасность и надёжность

### 5.1 Логирование и секреты

- В коде используются print и logger; в логах могут попадать цены и объёмы. Секреты API не логируются напрямую — хорошо.
- Файл `.env` в `.gitignore` — корректно.

---

### 5.2 Глобальный перехват исключений в LiveTrader

**Файл:** `src/trading/live_trader.py`, блок `finally`

```python
except:
    pass
```

При отмене ордеров исключения проглатываются без логирования. Лучше логировать и по возможности перебрасывать критичные ошибки.

---

### 5.3 Закрытие позиции при завершении LiveTrader

При завершении работы вызывается рыночный ордер на закрытие позиции. Это осознанное решение, но оно может быть нежелательным в части сценариев (например, при аварийном завершении). Стоит явно описать это в документации и при необходимости сделать опциональным (флаг «закрывать ли позицию при shutdown»).

---

## 6. Качество кода и поддерживаемость

### 6.1 Жёстко заданные пути и магические числа

- В `dataloader.calculate_features` путь к модулю фичей: `f"src/mlcore/features/{filename}.py"` — жёстко задан относительно текущей рабочей директории; при запуске из другой папки может не найтись файл.
- В коллекторах и трейдерах встречаются магические числа (например, 300 секунд warmup, 5 попыток reinit, 0.05 с интервал). Имеет смысл вынести их в конфиг или константы.

---

### 6.2 Неиспользуемые импорты

- **all_features.py / mid_features.py / spread_features.py:** импорты `erf, erfinv, gammaln, betainc, beta, gamma, digamma` из `scipy.special` и частично `stats` из `scipy` могут быть не использованы — стоит удалить неиспользуемое.
- **nn_features.py:** файл почти пустой (только импорты и пустые строки) — либо реализовать, либо убрать из репозитория/импортов.

---

### 6.3 README

- **README.md** содержит только заголовок «AlgoTradingSystem_c1». Нет описания назначения, зависимостей, способа запуска (collector, обучение, backtest, demo, live), переменных окружения и структуры проекта.

---

## 7. Резюме рекомендаций

| Приоритет | Что сделать |
|-----------|-------------|
| Критично   | Добавить или восстановить модули `mlcore.config`, `mlcore.grid`, `mlcore.stream_state`, `mlcore.predict_fast` (или убрать зависимость от них и переписать вызовы). |
| Критично   | Исправить импорт в `collector.py`: `from .storage import WALLogger`. |
| Критично   | Привести в соответствие сигнатуры процессоров и их вызовы в dataloader (убрать/добавить `canonical_grid`, `grid_config` по задумке архитектуры). |
| Критично   | Привести в соответствие сигнатуры процессоров и вызовы в dataloader (см. п. 1.4). |
| Критично   | Добавить в `requirements.txt` пакет `scikit-optimize` (или убрать использование `skopt` в backtest_trader). |
| Важно     | Задать `self.symbol` в `LiveTrader.__init__` или явно требовать вызов `run()` до использования. |
| Важно     | Унифицировать абстрактные методы трейдеров (async/signature) и их реализации в BacktestTrader/DemoTrader. |
| Желательно | Устранить дубликаты функций в all_features.py; добавить защиту от деления на ноль в herding_measure; проверять наличие колонок ask в orderbook_slope_5levels. |
| Желательно | Улучшить README, вынести ключевые параметры в конфиг, убрать неиспользуемые импорты. |

---

*Аудит выполнен по состоянию репозитория на момент проверки. После исправления критических пунктов имеет смысл прогнать сборку и тесты (если появятся) и повторить проверку.*
