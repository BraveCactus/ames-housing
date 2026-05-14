import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Заполняет пропуски в категориальных признаках и опционально кодирует их.
    """
    def __init__(self, cat_cols=None, encoding=True, fill_suffix="No"):
        """
        params:
            cat_cols: list - список названий категориальных колонок
            encoding: bool - нужно ли OneHot кодирование
            fill_suffix: str - префикс для заполнения пропусков
        """
        self.cat_cols = cat_cols
        self.encoding = encoding
        self.fill_suffix = fill_suffix
        self.ohe_encoder_ = None

    def fit(self, X, y=None):
        """
        Обучается на данных: запоминает категории для OneHot кодирования
        params:
            X: pd.DataFrame или np.array - входные данные для обучения
            y: pd.Series или np.array - целевая переменная
        """
        if isinstance(X, pd.DataFrame):
            self.cat_cols_ = self.cat_cols if self.cat_cols else X.columns.tolist()
        else:
            self.cat_cols_ = self.cat_cols

        X_filled = self._fill_missing(X)

        if self.encoding:
            self.ohe_encoder_ = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'
            )            
            if isinstance(X_filled, pd.DataFrame):
                self.ohe_encoder_.fit(X_filled[self.cat_cols_])
            else:
                self.ohe_encoder_.fit(X_filled)

        return self
    
    def transform(self, X):
        """
        Применяет заполнение и кодирование
        params:
            X: pd.DataFrame или np.array - входные данные для трансформации     
        returns:
            np.array - трансформированные данные
        """

        X_filled = self._fill_missing(X)

        if self.encoding:
            if isinstance(X_filled, pd.DataFrame):
                X_encoded = self.ohe_encoder_.transform(X_filled[self.cat_cols_])
            else:
                X_encoded = self.ohe_encoder_.transform(X_filled)
            return X_encoded

    def _fill_missing(self, X):
        """
        Внутренний метод для заполнения пропусков
        params:
            X: pd.DataFrame или np.array - входные данные для заполнения пропусков
        returns:    
            pd.DataFrame - данные с заполненными пропусками
        """
        
        if not isinstance(X, pd.DataFrame):
            if self.cat_cols_ is None:
                raise ValueError("cat_cols must be provided when X is numpy array")
            X = pd.DataFrame(X, columns=self.cat_cols_)
        
        X_filled = X.copy()
        for col in self.cat_cols_:
            X_filled[col] = X_filled[col].fillna(f"{self.fill_suffix}_{col}")
        
        return X_filled
    
    def get_feature_names_out(self, input_features=None):
        """Возвращает имена колонок после трансформации
        params:
            input_features: list - список входных признаков
        returns:
            list - список имен признаков после трансформации
        """
        if self.encoding and self.ohe_encoder_ is not None:
            return self.ohe_encoder_.get_feature_names_out(self.cat_cols_)
        return self.cat_cols_

class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Заполняет пропуски в числовых признаках и опционально масштабирует их.
    """
    def __init__(self, num_cols=None, impute_strategy='median', 
                 scaling=True, scaler_type='standard', fill_value=None):
        """
        params:
            num_cols: list - список названий числовых колонок
            impute_strategy: str - стратегия заполнения пропусков ('mean', 'median', 'constant', 'most_frequent')
            scaling: bool - нужно ли масштабирование
            scaler_type: str - тип масштабирования ('standard', 'minmax', 'robust')
            fill_value: any - значение для заполнения (если strategy='constant')
        """
        self.num_cols = num_cols
        self.impute_strategy = impute_strategy
        self.scaling = scaling
        self.scaler_type = scaler_type
        self.fill_value = fill_value
        self.imputer_ = None
        self.scaler_ = None
    
    def fit(self, X, y=None):
        """
        Обучается на данных: запоминает параметры для заполнения и масштабирования
        params:
            X: pd.DataFrame или np.array - входные данные для обучения
            y: pd.Series или np.array - целевая переменная
        """
        # Определяем колонки
        if isinstance(X, pd.DataFrame):
            self.num_cols_ = self.num_cols if self.num_cols else X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.num_cols_ = self.num_cols
            if self.num_cols_ is None:
                raise ValueError("num_cols must be provided when X is numpy array")        
        
        X_numeric = self._extract_numeric(X)        
        
        imputer_params = {'strategy': self.impute_strategy}
        if self.impute_strategy == 'constant' and self.fill_value is not None:
            imputer_params['fill_value'] = self.fill_value
        
        self.imputer_ = SimpleImputer(**imputer_params)
        self.imputer_.fit(X_numeric)        
        
        if self.scaling:
            if self.scaler_type == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler_ = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler_ = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {self.scaler_type}")            
            
            X_imputed = self.imputer_.transform(X_numeric)
            self.scaler_.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        """
        Применяет заполнение и масштабирование
        params:
            X: pd.DataFrame или np.array - входные данные для трансформации
        returns:
            X: np.array - трансформированные данные
        """
        X_numeric = self._extract_numeric(X)        
        
        X_imputed = self.imputer_.transform(X_numeric)        
        
        if self.scaling:
            X_scaled = self.scaler_.transform(X_imputed)
            return X_scaled
        
        return X_imputed
    
    def _extract_numeric(self, X):
        """
        Извлекает числовые колонки из DataFrame или numpy array
        params:
            X: pd.DataFrame или np.array - входные данные для извлечения числовых колонок
        returns:
            np.array - числовые данные для трансформации
        """
        if isinstance(X, pd.DataFrame):            
            missing_cols = [col for col in self.num_cols_ if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Колонки не найдены: {missing_cols}")
            return X[self.num_cols_].values
        else:            
            return X
    
    def get_feature_names_out(self, input_features=None):
        """Возвращает имена колонок после трансформации
        params:
            input_features: list - список входных признаков 
        returns:
            list - список имен признаков после трансформации"""
        if self.scaling:
            return [f"{col}_scaled" for col in self.num_cols_]
        return self.num_cols_
    
    def get_params(self, deep=True):
        """Возвращает параметры для GridSearch"""
        return {
            'num_cols': self.num_cols,
            'impute_strategy': self.impute_strategy,
            'scaling': self.scaling,
            'scaler_type': self.scaler_type,
            'fill_value': self.fill_value
        }
    
    def set_params(self, **params):
        """Устанавливает параметры для GridSearch"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def column_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Создает ColumnTransformer для числовых и категориальных признаков.
    params:
        num_cols: list - список названий числовых признаков для стандартизации
        cat_cols: list - список названий категориальных признаков для кодирования
    return:
        ColumnTransformer - преобразователь столбцов для стандартизации числовых и кодирования категориальных признаков
    """
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
