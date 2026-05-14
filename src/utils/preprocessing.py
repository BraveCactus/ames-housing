import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def impute_numerical_data(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в числовых признаках медианой и возвращает новый DataFrame ТОЛЬКО с заполненными числовыми данными.
    params:
        df: pd.DataFrame - входной DataFrame с числовыми признаками
        num_cols: list - список названий числовых признаков для заполнения
    return: 
        pd.DataFrame - новый DataFrame ТОЛЬКО с заполненными числовыми признаками
    """
    inputed_df = df.copy()
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(inputed_df[num_cols])
    imputed_df = pd.DataFrame(imputed_data, columns=num_cols, index=inputed_df.index)
    return imputed_df


def fill_categorical_data(df: pd.DataFrame, cat_cols: list, encoding: bool = False) -> pd.DataFrame:
    """
    Заполняет пропущенные значения в категориальных признаках и возвращает новый DataFrame ТОЛЬКО с закодированными признаками.
    params:
        df: pd.DataFrame - входной DataFrame с категориальными признаками
        cat_cols: list - список названий категориальных признаков для заполнения
        encoding: bool - флаг, указывающий на необходимость кодирования категориальных признаков
    return:
        pd.DataFrame - новый DataFrame ТОЛЬКО с закодированными категориальными признаками
    """
    filled_df = df.copy()
    for col in cat_cols:
        filled_df[col] = filled_df[col].fillna(f"No_{col}", inplace=False)

    if encoding:
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', 
                                    categories='auto',
                                    sparse_output=False,
                                    drop='first')
        encoded_data = ohe_encoder.fit_transform(filled_df[cat_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe_encoder.get_feature_names_out(cat_cols))
        return encoded_df

    return filled_df[cat_cols]