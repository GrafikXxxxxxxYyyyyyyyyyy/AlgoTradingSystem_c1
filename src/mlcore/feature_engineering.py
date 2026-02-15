"""
Feature engineering: вычисление корреляций признаков с таргетом (mid / spread).

Итеративный workflow:
  1. Добавляете новые функции-кандидаты в mid_features.py (или spread_features.py).
  2. Запускаете скрипт → получаете таблицу корреляций с таргетом.
  3. Признаки с |corr| выше порога (--min-corr) или топ-N по |corr| (--top) переносите
     в all_features.py — они будут использоваться в обучении и трейдинге.
  4. Очищаете mid_features.py (или spread_features), придумываете новую партию кандидатов,
     снова запускаете скрипт → получаете новые корреляции → сильнейшие снова переносите в all_features.
  5. Повторяете, пока не наберёте нужный набор признаков или не выйдете на желаемый уровень корреляции.

Итого: mid_features / spread_features — песочница для кандидатов; all_features.py — итоговый набор для пайплайна.
"""
from __future__ import annotations

import argparse
import inspect
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    mutual_info_regression = None

# Добавляем корень проекта в path для get_processed_data
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_FEATURES_DIR = Path(__file__).resolve().parent / "features"


def calculate_mid_target(df: pd.DataFrame) -> pd.Series:
    """Таргет для mid-модели: log(mid_next / close). Как в train_models."""
    target = np.zeros(len(df), dtype=float)
    for i in range(len(df) - 1):
        current = df["close"].iloc[i]
        mid = (df["high"].iloc[i + 1] + df["low"].iloc[i + 1]) / 2.0
        target[i] = np.log(mid / current) if current > 0 else 0.0
    return pd.Series(target, index=df.index)


def calculate_spread_target(df: pd.DataFrame) -> pd.Series:
    """Таргет для spread-модели: (high_next - low_next) / close. Как в train_models."""
    target = np.zeros(len(df), dtype=float)
    for i in range(len(df) - 1):
        current = df["close"].iloc[i]
        spread = df["high"].iloc[i + 1] - df["low"].iloc[i + 1]
        target[i] = spread / current if current > 0 else 0.0
    return pd.Series(target, index=df.index)


def load_feature_functions(module_name: str) -> dict:
    """Загружает все функции из src/mlcore/features/{module_name}.py."""
    path = _FEATURES_DIR / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Feature module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        name: func
        for name, func in inspect.getmembers(mod, inspect.isfunction)
    }


def build_feature_dataframe(raw: pd.DataFrame, feature_functions: dict) -> pd.DataFrame:
    """Строит DataFrame: каждая колонка — результат одной feature-функции."""
    f_dict = {}
    for name, func in tqdm(feature_functions.items(), desc="Features"):
        try:
            f_dict[name] = func(raw)
        except Exception as e:
            print(f"  → ERROR in {name}: {e}")
            f_dict[name] = pd.Series(0.0, index=raw.index)
    return pd.DataFrame(f_dict, index=raw.index)


def compute_correlations(
    df: pd.DataFrame,
    target_col: str = "target",
    use_mi: bool = True,
) -> pd.DataFrame:
    """
    Для каждой колонки (кроме target) считает Pearson, Spearman (IC), опционально MI.
    Возвращает DataFrame с колонками: feature, corr, p_value, ic, [mi].
    """
    t = df[target_col]
    result = []
    skip = {target_col, "price"}
    cols = [c for c in df.columns if c not in skip]

    for col in tqdm(cols, desc="Correlations"):
        feature = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        valid = np.isfinite(feature.values) & np.isfinite(t.values)
        if valid.sum() < 10:
            result.append({
                "feature": col,
                "corr": np.nan,
                "p_value": np.nan,
                "ic": np.nan,
                "mi": np.nan,
            })
            continue
        x, y = feature.values[valid], t.values[valid]
        corr, p_value = pearsonr(x, y)
        ic, _ = spearmanr(x, y)
        mi = np.nan
        if use_mi and mutual_info_regression is not None:
            try:
                mi_arr = mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=[False],
                    random_state=42,
                )
                mi = float(mi_arr[0])
            except Exception:
                pass
        result.append({
            "feature": col,
            "corr": corr,
            "p_value": p_value,
            "ic": ic,
            "mi": mi,
        })

    out = pd.DataFrame(result)
    out = out.sort_values("corr", key=lambda x: x.abs(), ascending=False)
    return out.reset_index(drop=True)


def run_feature_engineering(
    target_type: str = "mid",
    raw: pd.DataFrame | None = None,
    data_source: str = "local",
    symbol: str = "BTCUSDC",
    min_abs_correlation: float = 0.0,
    use_mi: bool = True,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Основная функция: загружает данные (если raw не передан), считает таргет,
    строит фичи из mid_features / spread_features, считает корреляции.

    Returns:
        feature_df: DataFrame с target и всеми фичами
        correlations: DataFrame с колонками feature, corr, p_value, ic, mi
        selected: список имён признаков с |corr| >= min_abs_correlation (или top_n лучших)
    """
    if raw is None:
        from src.mlcore.dataloader import get_processed_data
        raw = get_processed_data(
            source=data_source,
            symbol=symbol,
        )
    if raw.empty or len(raw) < 2:
        raise ValueError("Not enough data in raw DataFrame")

    module_name = "mid_features" if target_type == "mid" else "spread_features"
    if target_type == "mid":
        target = calculate_mid_target(raw)
    else:
        target = calculate_spread_target(raw)

    feature_functions = load_feature_functions(module_name)
    feature_df = build_feature_dataframe(raw, feature_functions)
    feature_df.insert(0, "target", target)

    correlations = compute_correlations(feature_df, target_col="target", use_mi=use_mi)

    if top_n is not None:
        selected = correlations.head(top_n)["feature"].tolist()
    else:
        selected = correlations[
            correlations["corr"].abs() >= min_abs_correlation
        ]["feature"].tolist()

    return feature_df, correlations, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering: корреляции признаков с таргетом (mid/spread). "
                    "Добавляйте функции в mid_features.py / spread_features.py и запускайте итеративно."
    )
    parser.add_argument(
        "--target",
        choices=("mid", "spread"),
        default="mid",
        help="Таргет: mid или spread",
    )
    parser.add_argument(
        "--data-source",
        default="local",
        choices=("local", "raw"),
        help="Источник данных для get_processed_data",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDC",
        help="Символ",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.0,
        help="Минимальный |corr| для отбора признаков (по умолчанию выводим все)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Вместо min-corr: оставить только top N признаков по |corr|",
    )
    parser.add_argument(
        "--no-mi",
        action="store_true",
        help="Не считать mutual information",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Сохранить таблицу корреляций в CSV",
    )
    parser.add_argument(
        "--out-selected",
        default=None,
        help="Сохранить список отобранных признаков (один на строку) в файл",
    )
    args = parser.parse_args()

    print(f"Target: {args.target}, data: {args.data_source}, min_abs_corr={args.min_corr}, top={args.top}")
    feature_df, correlations, selected = run_feature_engineering(
        target_type=args.target,
        raw=None,
        data_source=args.data_source,
        symbol=args.symbol,
        min_abs_correlation=args.min_corr,
        use_mi=not args.no_mi,
        top_n=args.top,
    )

    print("\n--- Корреляции (отсортированы по |corr|) ---")
    print(correlations.to_string())
    print(f"\n--- Признаков с |corr| >= {args.min_corr}: {len(selected)} ---")
    for name in selected[:50]:
        print(f"  {name}")
    if len(selected) > 50:
        print(f"  ... и ещё {len(selected) - 50}")

    if args.out_csv:
        correlations.to_csv(args.out_csv, index=False)
        print(f"\nКорреляции сохранены: {args.out_csv}")
    if args.out_selected:
        Path(args.out_selected).write_text("\n".join(selected), encoding="utf-8")
        print(f"Список признаков сохранён: {args.out_selected}")


if __name__ == "__main__":
    main()
