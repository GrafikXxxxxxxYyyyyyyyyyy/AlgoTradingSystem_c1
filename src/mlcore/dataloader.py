import os
import glob
import inspect
import numpy as np
import pandas as pd
import importlib.util

from tqdm import tqdm
from typing import Optional
from src.parser.live_collector import LiveCollector
from src.mlcore.processors import (
    BaseProcessor,
    AggTradesProcessor,
    RawTradesProcessor,
    OrderbookProcessor,
)



def process_raw_stream(
    stream_type: str,
    processor: BaseProcessor,
    symbol: str = "BTCUSDT",
    save_dir: str = "features/",
    drop_last: bool = True,
) -> pd.DataFrame:
    # Получаем все файлы в директории
    pattern = os.path.join(f"data/data_{symbol}", f"{stream_type}/{symbol}_{stream_type}_*.parquet")
    files = sorted(glob.glob(pattern))
    print(f"Найдено {len(files)} файлов для {stream_type}")

    # Опционально дропаем последний час
    if drop_last and len(files) > 0:
        files = files[:-1]
    
    df_list = []
    for f in tqdm(files):
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
    
    # Объединяем файлы и чистим
    stream_df = pd.concat(df_list, ignore_index=True)
    stream_df.drop_duplicates(subset='exchange_ts', keep='last', inplace=True)
    stream_df.dropna(inplace=True)
    
    # Сохраняем итоговый результат
    path_to_save = os.path.join(save_dir, f"{stream_type}.parquet")
    stream_df.to_parquet(path_to_save)

    return stream_df



def get_processed_data(
    source: str = "local", 
    symbol: str = "BTCUSDT",
    drop_last: bool = True,
    save_dir: str = "parquets/",
    live_collector: Optional[LiveCollector] = None,
):  
    # Инициализируем стримы и их процессоры 
    processors = {
        "aggTrades": AggTradesProcessor(),
        "rawTrades": RawTradesProcessor(),
        "orderbook_snapshots": OrderbookProcessor(),
    }
    
    df_features = []

    # Получаем данные по каждому из потоков
    if source == "raw":
        for stream_type, processor in processors.items():
            stream_df = process_raw_stream(
                stream_type=stream_type,
                processor=processor,
                symbol=symbol,
                save_dir=save_dir,
                drop_last=drop_last,
            )
            df_features.append(stream_df)

    elif source == "local":
        for stream_type, processor in processors.items():
            path_to_load = os.path.join(save_dir, f"{stream_type}.parquet")
            stream_df = pd.read_parquet(path_to_load)
            df_features.append(stream_df)

    elif source == "live":
        if live_collector is None:
            raise ValueError(f"LiveParser should be passed!")

        for stream_type, processor in processors.items():
            live_raw = live_collector.get_dataframe(stream_type)
            live_stream_df = processor(live_raw, need_prev_hour=False)
            live_stream_df.drop_duplicates(subset='exchange_ts', keep='last', inplace=True)
            live_stream_df.dropna(inplace=True)
            df_features.append(live_stream_df)

    else:
        raise ValueError(f"Unknown source: {source}")
    
    # Мержим всё в один датафрейм
    df_merged = df_features[0]
    for df in df_features[1:]:
        df_merged = pd.merge(df_merged, df, on='exchange_ts', how='inner')

    return df_merged



def calculate_features(raw: pd.DataFrame, filename="all_features"):
    spec = importlib.util.spec_from_file_location(f"{filename}", f"src/mlcore/features/{filename}.py")
    test_features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_features)

    feature_functions = {
        name: func
        for name, func in inspect.getmembers(test_features, inspect.isfunction)
    }
    
    f_dict = {}
    for name, func in feature_functions.items():
        try:
            f_dict[name] = func(raw)
        except Exception as e:
            print(f"  → ERROR in {name}: {str(e)}")

    df = pd.DataFrame(f_dict)
    
    return df
