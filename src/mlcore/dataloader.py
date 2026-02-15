"""
Загрузка и объединение потоков по единой временной сетке.
Поддержка тиковой/высокочастотной дискретизации (50–100 ms) без потери данных.
Данные только из parquet: local (merged.parquet) или raw (parquet по потокам в data/data_{symbol}/).
"""
import os
import glob
import inspect
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, Any

# Чтобы различать "параметр не передан" и "передан None" (тиковый режим)
_RESOLUTION_UNSET: Any = object()

from src.parser.live_collector import LiveCollector
from .processors import (
    BaseProcessor,
    AggTradesProcessor,
    RawTradesProcessor,
    OrderbookProcessor,
)


def process_raw_stream(
    stream_name: str,
    processor: BaseProcessor,
    symbol: str = "BTCUSDT",
    data_dir: str = "data",
    save_dir: str = "features/",
    drop_last: bool = True,
) -> pd.DataFrame:
    """Читает parquet-файлы потока из data_dir/data_{symbol}/{stream_name}/ и обрабатывает процессором."""
    base = os.path.join(data_dir, f"data_{symbol}", stream_name)
    files = sorted(glob.glob(os.path.join(base, f"{symbol}_{stream_name}_*.parquet")))
    if not files:
        return pd.DataFrame()
    if drop_last and len(files) > 0:
        files = files[:-1]

    df_list = []
    for f in tqdm(files, desc=stream_name):
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            df_processed = processor(df, need_prev_hour=True)
            df_list.append(df_processed)
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {f}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()
    stream_df = pd.concat(df_list, ignore_index=True)
    stream_df = stream_df.drop_duplicates(subset="exchange_ts", keep="last")
    return stream_df


def get_processed_data(
    source: str = "local",
    symbol: str = "BTCUSDT",
    drop_last: bool = True,
    save_dir: str = "parquets/",
    data_dir: str = "data",
    live_collector: Optional[LiveCollector] = None,
    grid_resolution_ms: Any = _RESOLUTION_UNSET,
) -> pd.DataFrame:
    """
    Возвращает объединённый датафрейм по всем потокам (aggTrades, rawTrades, orderbook) на единой сетке.

    source: "local" — merged.parquet из save_dir; "raw" — parquet из data_dir/data_{symbol}/; "live" — из live_collector.
    grid_resolution_ms: интервал сетки в мс (100, 5000, …). None — тиковый режим. По умолчанию 5000 (5 s).
    Потоки объединяются через outer merge по exchange_ts — строки не теряются (все метки времени сохраняются).
    """
    resolution_ms = grid_resolution_ms if grid_resolution_ms is not _RESOLUTION_UNSET else 5000

    processors = {
        "aggTrades": AggTradesProcessor(grid_resolution_ms=resolution_ms),
        "rawTrades": RawTradesProcessor(grid_resolution_ms=resolution_ms),
        "orderbook_snapshots": OrderbookProcessor(grid_resolution_ms=resolution_ms),
    }

    if source == "raw":
        df_features = []
        for stream_name, processor in processors.items():
            stream_df = process_raw_stream(
                stream_name=stream_name,
                processor=processor,
                symbol=symbol,
                data_dir=data_dir,
                save_dir=save_dir,
                drop_last=drop_last,
            )
            df_features.append(stream_df)

        if not df_features or all(d.empty for d in df_features):
            return pd.DataFrame()

        # Outer merge: keep every timestamp from every stream (no data loss)
        out = df_features[0]
        for df in df_features[1:]:
            if df.empty:
                continue
            out = pd.merge(out, df, on="exchange_ts", how="outer", suffixes=("", "_y"))
            out = out[[c for c in out.columns if not c.endswith("_y")]]
        out = out.sort_values("exchange_ts").drop_duplicates(subset="exchange_ts", keep="last")
        out = out.ffill()
        os.makedirs(save_dir, exist_ok=True)
        out.to_parquet(os.path.join(save_dir, "merged.parquet"))
        for stream_name, d in zip(processors.keys(), df_features):
            if not d.empty:
                d.to_parquet(os.path.join(save_dir, f"{stream_name}.parquet"))
        return out

    elif source == "local":
        path_merged = os.path.join(save_dir, "merged.parquet")
        if os.path.isfile(path_merged):
            return pd.read_parquet(path_merged)
        # Fallback: по одному потоку
        df_features = []
        for stream_name in processors:
            path_to_load = os.path.join(save_dir, f"{stream_name}.parquet")
            if os.path.isfile(path_to_load):
                df_features.append(pd.read_parquet(path_to_load))
        if not df_features:
            return pd.DataFrame()
        out = df_features[0]
        for df in df_features[1:]:
            out = pd.merge(out, df, on="exchange_ts", how="outer", suffixes=("", "_y"))
            out = out[[c for c in out.columns if not c.endswith("_y")]]
        return out.sort_values("exchange_ts").ffill()

    elif source == "live":
        if live_collector is None:
            raise ValueError("LiveParser should be passed!")

        df_features = []
        for stream_name, processor in processors.items():
            raw = live_collector.get_dataframe(stream_name)
            if raw.empty:
                continue
            proc = processor(raw, need_prev_hour=False)
            proc = proc.drop_duplicates(subset="exchange_ts", keep="last")
            df_features.append(proc)

        if not df_features:
            return pd.DataFrame()
        out = df_features[0]
        for df in df_features[1:]:
            out = pd.merge(out, df, on="exchange_ts", how="outer", suffixes=("", "_y"))
            out = out[[c for c in out.columns if not c.endswith("_y")]]
        return out.sort_values("exchange_ts").ffill()

    else:
        raise ValueError(f"Unknown source: {source}")



def calculate_features(raw: pd.DataFrame, filename: str = "all_features") -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        filename, f"src/mlcore/features/{filename}.py"
    )
    if spec is None or spec.loader is None:
        return pd.DataFrame()
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    feature_functions = {
        name: func
        for name, func in inspect.getmembers(mod, inspect.isfunction)
    }
    f_dict = {}
    for name, func in feature_functions.items():
        try:
            f_dict[name] = func(raw)
        except Exception as e:
            print(f"  → ERROR in {name}: {str(e)}")
    return pd.DataFrame(f_dict)
