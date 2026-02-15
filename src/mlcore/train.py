import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight



def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 1000,
    max_depth: int = 6,
    learning_rate: float = 0.01,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
    objective: str = 'reg:squaredlogerror', # или 'reg:squarederror'
    eval_metric: str = 'rmse', # или 'mae', 'mape'
    path_to_save: str = 'models/',
    model_name: str = "regression_xgb",
):
    # Обучение
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=early_stopping_rounds,
        objective=objective,
        eval_metric=eval_metric,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print("Обучение модели...")
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=50
    )

    save_path = os.path.join(path_to_save, f'{model_name}.json')
    model._estimator_type = "regressor"
    model.save_model(save_path)
    print(f"Модель сохранена в директорию: {save_path}")

    return model



def train_binary_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 1000,
    max_depth: int = 3,
    learning_rate: float = 0.01,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
    objective: str = 'binary:logistic',
    eval_metric: str = 'logloss',
    path_to_save: str = 'models/',
    model_name: str = "binary_xgb",
):
    model = XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method='hist',        
        early_stopping_rounds=early_stopping_rounds,
    )

    # Вычисляем веса для балансировки классов
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Обучаем модель с весами
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    save_path = os.path.join(path_to_save, f'{model_name}.json')
    model._estimator_type = 'classifier'
    model.save_model(save_path)
    print(f"Модель сохранена в директорию: {save_path}")

    return model