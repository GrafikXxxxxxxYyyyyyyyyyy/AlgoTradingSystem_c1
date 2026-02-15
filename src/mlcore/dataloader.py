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


def _rule_ns_from_resolution(resolution_kw: dict) -> Optional[int]:
    """Шаг сетки в наносекундах из resolution_kw (grid_resolution_ms); None = тиковый режим."""
    v = resolution_kw.get("grid_resolution_ms")
    return int(v * 1_000_000) if v is not None else None


def _reindex_to_regular_grid(df: pd.DataFrame, rule_ns: int) -> pd.DataFrame:
    """Приводит кадр к регулярной временной сетке с шагом rule_ns (одна строка на интервал)."""
    ts = pd.to_datetime(df["exchange_ts"])
    ts_min_ns = ts.min().value
    ts_max_ns = ts.max().value
    first_ns = (ts_min_ns // rule_ns) * rule_ns
    regular_ns = np.arange(first_ns, ts_max_ns + 1, rule_ns, dtype=np.int64)
    regular_index = pd.to_datetime(regular_ns, unit="ns")
    df = df.set_index("exchange_ts")
    df = df.reindex(regular_index).ffill()
    df = df.reset_index().rename(columns={"index": "exchange_ts"})
    return df


def process_raw_stream(
    stream_name: str,
    processor: BaseProcessor,
    symbol: str = "BTCUSDT",
    data_dir: str = "data",
    save_dir: str = "features/",
    drop_last: bool = True,
) -> pd.DataFrame:
    """
    Читает parquet-файлы потока и обрабатывает процессором.
    При заданной сетке (rule_ns): каждый файл обрабатывается с минимальным перекрытием
    (хвост предыдущего + текущий + голова следующего), затем оставляются только строки,
    принадлежащие текущему файлу. Данные на стыках не теряются, каждый файл читается один раз.
    """
    base = os.path.join(data_dir, f"data_{symbol}", stream_name)
    files = sorted(glob.glob(os.path.join(base, f"{symbol}_{stream_name}_*.parquet")))
    if not files:
        return pd.DataFrame()
    if drop_last and len(files) > 0:
        files = files[:-1]

    rule_ns = getattr(processor, "_rule_ns", None)

    # Тиковый режим или процессор без rule_ns — прежняя логика (prev file + current)
    if rule_ns is None:
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

    # Режим сетки: по одному файлу, с перекрытием tail_prev + current + head_next
    delta_ns = pd.Timedelta(nanoseconds=rule_ns)
    delta_2ns = pd.Timedelta(nanoseconds=2 * rule_ns)
    df_list = []
    prev_tail: Optional[pd.DataFrame] = None
    next_df: Optional[pd.DataFrame] = None

    for i in tqdm(range(len(files)), desc=stream_name):
        last_ts = None
        try:
            if i == 0:
                current_df = pd.read_parquet(files[0])
            else:
                current_df = next_df
            if current_df.empty:
                if i < len(files) - 1:
                    next_df = pd.read_parquet(files[i + 1])
                continue

            current_df = current_df.copy()
            current_df["exchange_ts"] = pd.to_datetime(current_df["exchange_ts"], unit="ms")
            first_ts = current_df["exchange_ts"].min()
            last_ts = current_df["exchange_ts"].max()

            if i < len(files) - 1:
                next_df = pd.read_parquet(files[i + 1])
                next_df = next_df.copy()
                next_df["exchange_ts"] = pd.to_datetime(next_df["exchange_ts"], unit="ms")
                next_head = next_df[next_df["exchange_ts"] <= last_ts + delta_ns]
            else:
                next_head = None

            if prev_tail is not None:
                prev_tail = prev_tail[prev_tail["exchange_ts"] >= first_ts - delta_ns]
                combined = pd.concat([prev_tail, current_df], ignore_index=True)
            else:
                combined = current_df
            if next_head is not None and not next_head.empty:
                combined = pd.concat([combined, next_head], ignore_index=True)

            t_keep_lo = first_ts + delta_ns
            t_keep_hi = last_ts + (delta_ns if next_head is not None and not next_head.empty else pd.Timedelta(0))

            df_processed = processor(
                combined,
                need_prev_hour=False,
                keep_ts_range=(t_keep_lo, t_keep_hi),
            )
            if not df_processed.empty:
                df_list.append(df_processed)

            prev_tail = current_df[current_df["exchange_ts"] >= last_ts - delta_2ns].copy()
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {files[i]}: {e}")
            try:
                if not current_df.empty and last_ts is not None:
                    prev_tail = current_df[current_df["exchange_ts"] >= last_ts - delta_2ns].copy()
            except NameError:
                pass
            if i < len(files) - 1 and next_df is None:
                next_df = pd.read_parquet(files[i + 1])
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
    grid_resolution_ms: интервал сетки в мс (5000 = 5 s, 10000 = 10 s, 100 = 100 ms). None — тиковый режим. По умолчанию 5000.
    Потоки объединяются через outer merge по exchange_ts — строки не теряются (все метки времени сохраняются).
    """
    if grid_resolution_ms is not _RESOLUTION_UNSET:
        resolution_kw = {"grid_resolution_ms": grid_resolution_ms}
    else:
        resolution_kw = {"grid_resolution_ms": 5000}

    processors = {
        "aggTrades": AggTradesProcessor(**resolution_kw),
        "rawTrades": RawTradesProcessor(**resolution_kw),
        "orderbook_snapshots": OrderbookProcessor(**resolution_kw),
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
        # Приводим к регулярной сетке с шагом rule_ns (50 ms, 5000 ms и т.д.), чтобы не было скачков 50/100 ms
        rule_ns = _rule_ns_from_resolution(resolution_kw)
        if rule_ns is not None and not out.empty:
            out = _reindex_to_regular_grid(out, rule_ns)
        os.makedirs(save_dir, exist_ok=True)
        out.to_parquet(os.path.join(save_dir, "merged.parquet"))
        for stream_name, d in zip(processors.keys(), df_features):
            if not d.empty:
                d.to_parquet(os.path.join(save_dir, f"{stream_name}.parquet"))
        return out

    elif source == "local":
        path_merged = os.path.join(save_dir, "merged.parquet")
        if os.path.isfile(path_merged):
            out = pd.read_parquet(path_merged)
        else:
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
            out = out.sort_values("exchange_ts").ffill()
        rule_ns = _rule_ns_from_resolution(resolution_kw)
        if rule_ns is not None and not out.empty:
            out = _reindex_to_regular_grid(out, rule_ns)
        return out

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
        out = out.sort_values("exchange_ts").ffill()
        rule_ns = _rule_ns_from_resolution(resolution_kw)
        if rule_ns is not None and not out.empty:
            out = _reindex_to_regular_grid(out, rule_ns)
        return out

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
