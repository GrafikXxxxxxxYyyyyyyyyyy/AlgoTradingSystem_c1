import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from scipy import stats
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def regression_metrics(y_true, y_pred, y_base=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mean_abs_target = np.mean(np.abs(y_true))
    mae_ratio = mae / mean_abs_target if mean_abs_target > 0 else np.nan

    if y_base is not None:
        y_base = np.asarray(y_base)
        rmse_base = np.sqrt(mean_squared_error(y_true, y_base))
        rmse_ratio = rmse / rmse_base if rmse_base > 0 else np.nan
    else:
        rmse_ratio = None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "MAE / mean|Δ|": mae_ratio,
        "RMSE / RMSE_base": rmse_ratio,
        "mean|Δ| (target)": mean_abs_target
    }



def directional_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    sign_acc = (np.sign(y_pred) == np.sign(y_true)).mean()
    pos_ratio = (y_true > 0).mean()
    
    y_true_cls = (y_true > 0).astype(int)
    y_pred_cls = (y_pred > 0).astype(int)
    
    correct_mask = (y_pred_cls == y_true_cls)
    incorrect_mask = ~correct_mask
    
    profit_correct = y_true[correct_mask].sum()
    loss_incorrect = -y_true[incorrect_mask].sum()
    profit_factor = profit_correct / (loss_incorrect + 1e-9)
    
    avg_gain_correct = y_true[correct_mask].mean() if correct_mask.any() else 0
    avg_loss_incorrect = y_true[incorrect_mask].mean() if incorrect_mask.any() else 0
    
    return {
        "Directional Accuracy": sign_acc,
        "Target Pos Ratio": pos_ratio,
        "Profit Factor": profit_factor,
        "Avg Gain (correct)": avg_gain_correct,
        "Avg Loss (incorrect)": avg_loss_incorrect
    }



def statistical_significance(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n = len(y_true)
    observed_acc = (np.sign(y_pred) == np.sign(y_true)).sum()
    
    p_binom = stats.binomtest(observed_acc, n=n, p=0.5, alternative='greater').pvalue
    
    boot_acc = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        acc = (np.sign(y_pred[idx]) == np.sign(y_true[idx])).mean()
        boot_acc.append(acc)

    ci_low, ci_high = np.percentile(boot_acc, [100*alpha/2, 100*(1-alpha/2)])
    
    return {
        "Binomial p-value (vs 50%)": p_binom,
        "Directional Acc 95% CI": (ci_low, ci_high),
        "Is Significant (α=5%)": p_binom < alpha
    }


def quantile_performance(y_true, y_pred, n_quantiles=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    confidence = np.abs(y_pred)
    try:
        quantiles = pd.qcut(confidence, q=n_quantiles, duplicates='drop')
    except ValueError:
        # Если все значения одинаковые
        quantiles = pd.cut(confidence, bins=n_quantiles, duplicates='drop')
    
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'quantile': quantiles,
        'confidence': confidence
    }).dropna()

    if df.empty:
        return pd.DataFrame(columns=['quantile', 'confidence', 'directional_acc', 'mean_return', 'n_samples'])
    
    result = []
    for q, group in df.groupby('quantile'):
        acc = (np.sign(group['y_pred']) == np.sign(group['y_true'])).mean()
        mean_conf = group['confidence'].mean()
        mean_return = group['y_true'].mean()
        count = len(group)
        result.append({
            'quantile': q,
            'confidence': mean_conf,
            'directional_acc': acc,
            'mean_return': mean_return,
            'n_samples': count
        })
    
    return pd.DataFrame(result).sort_values('confidence', ignore_index=True)



def plot_regression(y_true, y_pred, timestamps=None, figsize=(12, 8)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
    else:
        timestamps = np.arange(len(y_true))
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Scatter
    axes[0,0].scatter(y_pred, y_true, alpha=0.3, s=1)
    axes[0,0].axhline(0, color='k', linewidth=0.5)
    axes[0,0].axvline(0, color='k', linewidth=0.5)
    axes[0,0].set_xlabel('Prediction')
    axes[0,0].set_ylabel('True')
    axes[0,0].set_title('Prediction vs True')
    
    # 2. Error distribution
    error = y_pred - y_true
    axes[0,1].hist(error, bins=100, alpha=0.7)
    axes[0,1].set_title('Error Distribution')
    axes[0,1].set_xlabel('Error')
    
    # 3. Time series (sample)
    if timestamps is not None and len(timestamps) > 0:
        n_plot = min(5000, len(y_true))
        plot_idx = np.random.choice(len(y_true), size=n_plot, replace=False)
        axes[0,2].plot(timestamps[plot_idx], y_true[plot_idx], label='True', alpha=0.7, linewidth=0.8)
        axes[0,2].plot(timestamps[plot_idx], y_pred[plot_idx], label='Pred', alpha=0.7, linewidth=0.8)
        axes[0,2].set_title('True vs Pred (sample)')
        axes[0,2].legend()
        axes[0,2].set_xlabel('Time')
    else:
        axes[0,2].text(0.5, 0.5, 'No timestamps', ha='center', va='center')
        axes[0,2].set_title('True vs Pred')
    
    # 4. Confidence vs Accuracy
    qdf = quantile_performance(y_true, y_pred)
    if not qdf.empty:
        axes[1,0].plot(qdf['confidence'], qdf['directional_acc'], marker='o')
    axes[1,0].set_xlabel('Mean Confidence (|pred|)')
    axes[1,0].set_ylabel('Directional Accuracy')
    axes[1,0].set_title('Accuracy vs Confidence')
    axes[1,0].grid(True)
    
    # 5. Cumulative PnL
    pnl = np.sign(y_pred) * y_true
    cum_pnl = np.cumsum(pnl)
    axes[1,1].plot(cum_pnl)
    axes[1,1].set_title('Cumulative PnL (idealized)')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Cumulative log-return')
    
    # 6. PnL distribution
    axes[1,2].hist(pnl, bins=100, alpha=0.7)
    axes[1,2].axvline(0, color='k', linewidth=0.5)
    axes[1,2].set_title('PnL per Signal')
    
    plt.tight_layout()
    plt.show()
    
    return qdf



def evaluate_regression_model(
    y_test: pd.Series, 
    y_pred: pd.Series, 
    length: int = 1000,
    model_name: str = "XGBRegressor",
):
    print("=" * 60)
    
    # 1. Регрессионные метрики
    reg = regression_metrics(y_test, y_pred, y_base=np.zeros_like(y_test))
    for k, v in reg.items():
        if v is not None:
            if isinstance(v, float):
                print(f"{k:25s}: {v:.6f}")
            else:
                print(f"{k:25s}: {v}")
    
    print("\n" + "-" * 60)

    # 2. Направленческие метрики
    dir_metrics = directional_metrics(y_test, y_pred)
    for k, v in dir_metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.6f}")
        else:
            print(f"{k:25s}: {v}")
    
    print("\n" + "-" * 60)

    # 3. Статистическая значимость
    stat = statistical_significance(y_test, y_pred)
    for k, v in stat.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.6f}")
        elif isinstance(v, tuple):
            print(f"{k:30s}: ({v[0]:.4f}, {v[1]:.4f})")
        else:
            print(f"{k:30s}: {v}")
    
    print("\n" + "=" * 60)

    print("📊 Квантильный анализ (топ-3 квантиля по уверенности):")
    
    qdf = quantile_performance(y_test, y_pred)
    if not qdf.empty:
        print(qdf.sort_values('confidence', ascending=False).head(3).to_string(index=False))
    else:
        print("  → Недостаточно данных для квантильного анализа")
    
    # # 4. Визуализация
    # print("\n📈 Генерация визуализаций...")
    # qdf_full = plot_regression(y_test, y_pred)

    # 5. Сравнение графиков
    plt.figure(figsize=(24, 8))
    plt.plot(y_test.values[:length], label='Факт', alpha=0.7, color='black')
    plt.plot(y_pred[:length], label='Прогноз', alpha=0.7, color='red')
    plt.legend()
    plt.title(f"Прогноз vs Факт (первые {length} точек)")
    plt.show()



def evaluate_binary_classifier(
    y_test: pd.Series, 
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
):
    # === 1. Получите предсказания вашей обученной модели ===
    y_pred_thr = (y_pred_proba >= threshold).astype(int)

    # === 2. Основные метрики ===
    acc = accuracy_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr)
    rec = recall_score(y_test, y_pred_thr)
    f1 = f1_score(y_test, y_pred_thr)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print("📊 Основные метрики:")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"AUC-ROC:        {auc:.4f}")
    print(f"Log-Loss:       {logloss:.4f}")
    print("\n" + "="*50)
    print("Подробный отчёт (classification report):")
    print(classification_report(y_test, y_pred_thr))

    # === 3. Confusion Matrix ===
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred_thr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')

    # === 4. ROC-AUC Curve ===
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")

    # === 5. Распределение вероятностей по классам ===
    plt.subplot(1, 3, 3)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    proba_class_0 = y_pred_proba[y_test_np == 0]
    proba_class_1 = y_pred_proba[y_test_np == 1]

    plt.hist(proba_class_0, bins=100, alpha=0.7, label='Класс 0 (вниз)', color='red')
    plt.hist(proba_class_1, bins=100, alpha=0.7, label='Класс 1 (вверх)', color='green')
    plt.xlabel('Вероятность класса 1')
    plt.ylabel('Частота')
    plt.title('Распределение вероятностей по классам')
    plt.legend()

    plt.tight_layout()
    plt.show()