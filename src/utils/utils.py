import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Проверяет наличие пропущенных значений в DataFrame и возвращает серию с количеством пропущенных значений для каждого столбца.
    params:
        df: pd.DataFrame - входной DataFrame для проверки
    return:
        pd.Series - серия с количеством пропущенных значений для каждого столбца
    """
    missing_values = df.isnull().sum()    
    return missing_values

def visualize_categorical_features(df: pd.DataFrame, cat_cols: list) -> None:
    """
    Визуализирует распределение категориальных признаков с помощью гистограмм.
    params:
        df: pd.DataFrame - входной DataFrame для визуализации
        cat_cols: list - список названий категориальных признаков для визуализации    
    """
    n_cols = len(cat_cols)
    n_plots_per_row = 3
    n_rows = math.ceil(n_cols / n_plots_per_row)

    fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(18, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        df[col].value_counts().head(10).plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{col} (топ-10)')
        axes[i].tick_params(axis='x', rotation=45)

    # Скрываем лишние пустые графики
    for i in range(len(cat_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def visualize_numerical_features(df: pd.DataFrame, num_cols: list) -> None:
    """
    Визуализирует распределение числовых признаков с помощью боксплотов.
    params:
        df: pd.DataFrame - входной DataFrame для визуализации
        num_cols: list - список названий числовых признаков для визуализации    
    """
    n_cols = len(num_cols)
    n_plots_per_row = 3
    n_rows = math.ceil(n_cols / n_plots_per_row)

    fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(18, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        df[col].plot(kind='box', ax=axes[i])
        axes[i].set_title(col)

    # Скрываем лишние пустые графики
    for i in range(len(num_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, float, float]:
    """
    Находит выбросы в числовом столбце с помощью метода межквартильного размаха (IQR) и возвращает DataFrame с выбросами и границы.
    params:
        df: pd.DataFrame - входной DataFrame для анализа
        col: str - название числового столбца для поиска выбросов
    return:
        tuple[pd.DataFrame, float, float] - кортеж из DataFrame с выбросами, нижней и верхней границей
    """
    threshold = 1.5    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers, lower, upper

def show_metrics(y_preds, y_true): 
    y_mean = y_true.mean()  

    mae = mean_absolute_error(y_preds, y_true)
    rmse = np.sqrt(mean_squared_error(y_preds, y_true))

    relative_mae_error = round((mae / y_mean), 3)  
    relative_rmse_error = round((rmse / y_mean), 3)  

    print(f"MEAN:  ${y_mean:,.0f}")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RELATIVE MAE ERROR:  {relative_mae_error}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"RELATIVE RMSE ERROR:  {relative_rmse_error}")
    print(f"R²:   {r2_score(y_preds, y_true):.4f}")
