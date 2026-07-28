import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Заполняет пропущенные значения в числовых признаках
    """
    def __init__(self, 
                 num_cols=None, 
                 impute_strategy='simple_imputer',    
                 imputer_params: dict = dict(),             
                 ):
        self.num_cols = num_cols
        self.impute_strategy = impute_strategy # Возможные стратегии: simple_imputer, knn_imputer, iterative_imputer
        self.imputer_params =  imputer_params
        self.imputer_ = None 
        self.num_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Обучение на данных""" 
           
        if isinstance(X, pd.DataFrame):
            if self.num_cols is None:
                self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self.num_cols_ = self.num_cols
        else:
            print(f"X must be pandas DataFrame, got {type(X)}")

        missing_cols = [col for col in self.num_cols_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Колонки не найдены: {missing_cols}")

        X_numeric = X[self.num_cols]

        if self.impute_strategy == 'simple_imputer':
            self.imputer_ = SimpleImputer(**self.imputer_params)
        elif self.impute_strategy == 'knn_imputer':
            self.imputer_ = KNNImputer(**self.imputer_params)
        elif self.impute_strategy == 'iterative_imputer':
            self.imputer_ = IterativeImputer(**self.imputer_params)
        else:
            raise ValueError(
                f"Unknown impute strategy: {self.impute_strategy}"
                f"Available strategies: simple_imputer, knn_imputer, iterative_imputer"
            )
        
        self.imputer_.fit(X_numeric)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Заполняет пропущенные значения в данных"""
        if self.num_cols_ is None:
            raise RuntimeError("nImputer must be fitted before transform. Call fit() first.")
        X_numeric = X[self.num_cols_]
        X_numeric_imputed = self.imputer_.transform(X_numeric) # Возвращает только те столбцы, где были пропуски

        X_imputed = X.copy()
        X_imputed[self.num_cols] = X_numeric_imputed

        return X_imputed

class NumericalScaler(BaseEstimator, TransformerMixin):
    """Масштабирует числовые признаки"""
    def __init__(self, 
                 num_cols=None, 
                 scaler_type='standard_scaler', 
                 scaler_params: dict = dict()
                 ):
        
        self.num_cols = num_cols
        self.scaler_type = scaler_type # Возможные типы: standard_scaler, min_max_scaler, robust_scaler
        self.scaler_params = scaler_params
        self.scaler_ = None
        self.num_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Обучение на данных"""
        if isinstance(X, pd.DataFrame):
            if self.num_cols is None:
                self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self.num_cols_ = self.num_cols

        missing_cols = [col for col in self.num_cols_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Колонки не найдены: {missing_cols}")

        X_numeric = X[self.num_cols]

        if self.scaler_type == 'standard_scaler':
            self.scaler_ = StandardScaler(**self.scaler_params)
        elif self.scaler_type == 'min_max_scaler':
            self.scaler_ = MinMaxScaler(**self.scaler_params)
        elif self.scaler_type == 'robust_scaler':
            self.scaler_ = RobustScaler(**self.scaler_params)
        else:
            raise ValueError(
                f"Unknown scaler: {self.scaler_type}"
                f"Available scalers: standard_scaler, min_max_scaler, robust_scaler"
            )
        
        self.scaler_.fit(X_numeric)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Масштабирует числовые данные"""

        if self.num_cols_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        X_numeric = X[self.num_cols_]
        X_numeric_scaled = self.scaler_.transform(X_numeric) # Возващает только те столбцы, где были пропуски

        X_scaled = X.copy()
        X_scaled[self.num_cols] = X_numeric_scaled

        return X_scaled

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Заполняет пропущенные значения в категориальных признаках"""

    def __init__(self, 
                 cat_cols=None, 
                 filler_strategy='mode', 
                 filler_params: dict = dict(),                 
                ):
        self.cat_cols = cat_cols
        self.filler_strategy = filler_strategy # Возможные стратегии: mode, constant
        self.filler_params = filler_params
        self.filler_ = None
        self.cat_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Обучение на данных"""

        if isinstance(X, pd.DataFrame):
            if self.cat_cols is None:
                self.cat_cols_ = X.select_dtypes(include=[object]).columns.tolist()
            else:
                self.cat_cols_ = self.cat_cols

        missing_cols = [col for col in self.cat_cols_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Колонки не найдены: {missing_cols}")

        if self.filler_strategy == 'mode':
            self.filler_ = SimpleImputer(strategy='most_frequent')
        elif self.filler_strategy == 'constant':
            self.filler_ = SimpleImputer(**self.filler_params)
        else:
            raise ValueError(
                f"Unknown impute strategy: {self.filler_strategy}"
                f"Available strategies: mode, constant"
            )
        
        self.filler_.fit(X[self.cat_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Заполняет пропущенные значения в данных"""

        if self.cat_cols_ is None:
            raise RuntimeError("Imputer must be fitted before transform.")
        X_categorical = X[self.cat_cols_]
        X_categorical_imputed = self.filler_.transform(X_categorical) # Возвращает только те столбцы, где были пропуски

        X_imputed = X.copy()
        X_imputed[self.cat_cols_] = X_categorical_imputed

        return X_imputed

    def get_feature_names_out(self, input_features=None):
        return self.cat_cols_ if self.cat_cols_ is not None else []

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """"Кодирует категориальные признаки"""

    def __init__(self, 
                 cat_cols=None, 
                 encoder_strategy='one_hot_encoding', 
                 encoder_params: dict = dict(),                 
                ):
        self.cat_cols = cat_cols
        self.encoder_strategy = encoder_strategy # Возможные стратегии: one_hot_encoding, ordinal_encoding
        self.encoder_params = encoder_params
        self.encoder_ = None
        self.cat_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Обучение на данных"""

        if isinstance(X, pd.DataFrame):
            if self.cat_cols is None:
                self.cat_cols_ = X.select_dtypes(include=[object]).columns.tolist()
            else:
                self.cat_cols_ = self.cat_cols

        missing_cols = [col for col in self.cat_cols_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Колонки не найдены: {missing_cols}")    

        if self.encoder_strategy == 'one_hot_encoding':
            self.encoder_ = OneHotEncoder(**self.encoder_params)
        elif self.encoder_strategy == 'ordinal_encoding':
            self.encoder_ = OrdinalEncoder(**self.encoder_params)
        else:
            raise ValueError(
                f"Unknown encoder strategy: {self.encoder_strategy}"
                f"Available strategies: one_hot_encoding, ordinal_encoding"
            )

        self.encoder_.fit(X[self.cat_cols_])

        if self.encoder_strategy == 'one_hot_encoding':
            self.encoded_columns_ = self.encoder_.get_feature_names_out(self.cat_cols_)
        else:
            self.encoded_columns_ = self.cat_cols_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Кодирует категориальные признаки"""
        if self.cat_cols_ is None:
            raise RuntimeError("Encoder must be fitted before transform. Call fit() first.")
        
        X_categorical = X[self.cat_cols_]
        X_encoded = self.encoder_.transform(X_categorical)
        
        # !Для OneHotEncoder возвращается разреженная матрица!
        if self.encoder_strategy == 'one_hot_encoding':
            X_encoded = X_encoded.toarray()        
        
        X_encoded_df = pd.DataFrame(
            X_encoded,
            columns=self.encoded_columns_,
            index=X.index
        )        
        
        X_transformed = X.drop(columns=self.cat_cols_)
        X_transformed = pd.concat([X_transformed, X_encoded_df], axis=1)
        
        return X_transformed                          
                          

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
