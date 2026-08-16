from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.pipeline import Pipeline

from lightgbm import LGBMModel
from xgboost import XGBModel


class MLogger:
    """
    Класс для логирования ML-экспериментов с использованием MLflow.

    Поддерживает:
    - Обучение одной модели с фиксированными параметрами (run_single)
    - Поиск гиперпараметров: GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV (run_search)
    """

    def __init__(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        preprocessor: ColumnTransformer | None = None,
        tracking_uri: str | None = None,
    ):
        """
        Инициализирует логгер MLflow: задаёт URI сервера и имя эксперимента.

        params:
            experiment_name - имя эксперимента в MLflow.
            X_train - обучающие признаки.
            y_train - целевая переменная для обучения.
            X_test - тестовые признаки.
            y_test - целевая переменная для теста.
            preprocessor - трансформер, будет встроен в пайплайн.
            tracking_uri - URI сервера MLflow. Если передан,
                           вызывается mlflow.set_tracking_uri().
        """
        self.experiment_name = experiment_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _make_pipeline(self, model: BaseEstimator) -> BaseEstimator:
        """
        Собирает пайплайн из препроцессора и модели.

        params:
            model - модель.
        returns:
            пайплайн с препроцессором и моделью
            или модель без препроцессора.
        """
        if self.preprocessor is None:
            return model
        return Pipeline([("preprocessor", self.preprocessor), ("model", model)])

    def _log_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        prefix: str = "test",
    ) -> tuple[float, float, float]:
        """
        Вычисляет и логирует метрики MAE, RMSE, R².

        params:
            y_true - истинные значения.
            y_pred - предсказанные значения.
            prefix - префикс имени метрики ('train' или 'test').
        returns:
            кортеж из MAE, RMSE, R².
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mlflow.log_metrics(
            {f"{prefix}_mae": mae, f"{prefix}_rmse": rmse, f"{prefix}_r2": r2}
        )
        return mae, rmse, r2

    def _log_params_from_dict(self, params_dict: dict | None, prefix: str = "") -> None:
        """
        Логирует скалярные параметры из словаря с опциональным префиксом.

        Нескалярные значения (объекты оценщиков, списки и т.п.) пропускаются,
        чтобы не ломать mlflow.log_params() для ансамблей.

        params:
            params_dict - словарь параметров для логирования.
            prefix - префикс для имён параметров.
        """
        if not params_dict:
            return
        params_to_log = {}
        for key, value in params_dict.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                log_key = f"{prefix}{key}" if prefix else key
                params_to_log[log_key] = value
        if params_to_log:
            mlflow.log_params(params_to_log)

    def _log_model_params(self, model: BaseEstimator, pipeline: BaseEstimator | None) -> None:
        """
        Логирует эффективные гиперпараметры оценщика после set_params.

        params:
            model - исходная модель.
            pipeline - собранный пайплайн или модель.
        """
        if (
            pipeline is not None
            and hasattr(pipeline, "named_steps")
            and "model" in pipeline.named_steps
        ):
            estimator = pipeline.named_steps["model"]
        else:
            estimator = model
        try:
            params = estimator.get_params()
        except Exception:
            params = {}
        self._log_params_from_dict(params, prefix="")

    def _log_dataset_info(self) -> None:
        """Логирует информацию о данных: размеры, число признаков, список фич."""
        mlflow.log_params(
            {
                "n_train_samples": self.X_train.shape[0],
                "n_test_samples": self.X_test.shape[0],
                "n_features": self.X_train.shape[1],
            }
        )
        feature_list = getattr(self.X_train, "columns", None)
        if feature_list is not None:
            mlflow.log_param("feature_list", ",".join(feature_list.tolist()))

    def _needs_cloudpickle(self, estimator: BaseEstimator) -> bool:
        """
        Проверяет, требует ли оценщик cloudpickle-сериализации.

        skops (формат по умолчанию в mlflow 3.x) не умеет сериализовать
        LightGBM/XGBoost. Проверяем рекурсивно и ансамбли/пайплайны,
        содержащие такие модели.

        params:
            estimator - оценщик для проверки.
        returns:
            True, если модель требует cloudpickle, иначе False.
        """
        if isinstance(estimator, (LGBMModel, XGBModel)):
            return True
        if isinstance(estimator, Pipeline):
            return any(self._needs_cloudpickle(step) for _, step in estimator.steps)
        if isinstance(estimator, ColumnTransformer):
            return any(self._needs_cloudpickle(t) for _, t, _ in estimator.transformers)
        if isinstance(estimator, StackingRegressor):
            if self._needs_cloudpickle(estimator.final_estimator):
                return True
            return any(self._needs_cloudpickle(est) for _, est in estimator.estimators)
        if isinstance(estimator, VotingRegressor):
            return any(self._needs_cloudpickle(est) for _, est in estimator.estimators)
        return False

    def _log_model_artifact(self, pipeline: BaseEstimator, model: BaseEstimator) -> None:
        """
        Логирует модель в MLflow (артефакт 'model').

        params:
            pipeline - обученный пайплайн или модель.
            model - исходная модель (для тега model_type).
        """
        mlflow.set_tag("model_type", model.__class__.__name__)
        if self._needs_cloudpickle(model):
            mlflow.sklearn.log_model(
                pipeline, name="model", serialization_format="cloudpickle"
            )
        else:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "numpy.dtype[float64]",
                    "numpy.dtype[int64]",
                ],
            )

    def run_single(
        self,
        model: BaseEstimator,
        params: dict | None = None,
        run_name: str | None = None,
    ) -> BaseEstimator:
        """
        Обучает одну модель с заданными параметрами и логирует результаты.

        params:
            model - необученный оценщик (может быть уже с параметрами).
            params - параметры для модели (будут применены через set_params).
            run_name - имя запуска в MLflow.
        returns:
            обученный пайплайн или модель.
        """
        pipeline = self._make_pipeline(model)

        if params:
            pipeline.set_params(**params)

        run_name = run_name or f"{model.__class__.__name__}_single"

        with mlflow.start_run(run_name=run_name):
            self._log_dataset_info()
            self._log_model_params(model, pipeline)

            pipeline.fit(self.X_train, self.y_train)

            train_pred = pipeline.predict(self.X_train)
            test_pred = pipeline.predict(self.X_test)

            train_mae, train_rmse, train_r2 = self._log_metrics(
                self.y_train, train_pred, prefix="train"
            )
            test_mae, test_rmse, test_r2 = self._log_metrics(
                self.y_test, test_pred, prefix="test"
            )

            self._log_model_artifact(pipeline, model)

            print(f"   Запуск '{run_name}' завершён")
            print(f"   Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"   Train MAE: ${train_mae:,.0f}, Test MAE: ${test_mae:,.0f}")

            return pipeline

    def run_search(
        self,
        model: BaseEstimator,
        param_distributions: dict,
        search_type: str = "random",
        n_iter: int = 10,
        cv: int = 5,
        scoring: str = "neg_mean_absolute_error",
        random_state: int = 42,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[BaseEstimator, dict]:
        """
        Запускает поиск гиперпараметров с логированием в MLflow.

        params:
            model - необученный оценщик.
            param_distributions - словарь гиперпараметров для поиска.
            search_type - 'grid', 'random' или 'halving'.
            n_iter - число итераций (для random и halving).
            cv - число фолдов кросс-валидации.
            scoring - метрика для оптимизации.
            random_state - для воспроизводимости.
            run_name - имя запуска в MLflow.
            **kwargs - дополнительные параметры для поиска.
        returns:
            кортеж из лучшего оценщика и лучших параметров.
        """
        pipeline = self._make_pipeline(model)

        if search_type.lower() == "grid":
            searcher = GridSearchCV(
                pipeline,
                param_distributions,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                **kwargs,
            )
            search_name = "GridSearch"

        elif search_type.lower() == "random":
            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
                **kwargs,
            )
            search_name = "RandomSearch"

        elif search_type.lower() == "halving":
            searcher = HalvingRandomSearchCV(
                pipeline,
                param_distributions,
                n_candidates=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
                **kwargs,
            )
            search_name = "HalvingRandomSearch"

        else:
            raise ValueError("search_type must be 'grid', 'random' or 'halving'")

        run_name = run_name or f"{model.__class__.__name__}_{search_name}"

        with mlflow.start_run(run_name=run_name):
            self._log_dataset_info()
            mlflow.set_tag("model_type", model.__class__.__name__)
            mlflow.set_tag("search_type", search_name.lower())
            mlflow.log_params(
                {
                    "search_type": search_name.lower(),
                    "cv": cv,
                    "scoring": scoring,
                    "n_iter": n_iter if search_type != "grid" else "N/A",
                    "random_state": random_state if search_type != "grid" else "N/A",
                }
            )

            mlflow.log_dict(
                {k: str(v) for k, v in param_distributions.items()},
                "param_distributions.json",
            )

            searcher.fit(self.X_train, self.y_train)

            best_params = searcher.best_params_
            self._log_params_from_dict(best_params, prefix="best_")

            if search_type.lower() == "halving":
                mlflow.log_params(
                    {
                        "n_resources": searcher.n_resources_,
                        "n_candidates": searcher.n_candidates_,
                        "n_remaining_candidates": searcher.n_remaining_candidates_,
                        "max_resources": searcher.max_resources_,
                    }
                )

            train_pred = searcher.predict(self.X_train)
            test_pred = searcher.predict(self.X_test)

            train_mae, train_rmse, train_r2 = self._log_metrics(
                self.y_train, train_pred, prefix="train"
            )
            test_mae, test_rmse, test_r2 = self._log_metrics(
                self.y_test, test_pred, prefix="test"
            )

            mlflow.log_table(
                pd.DataFrame(searcher.cv_results_), artifact_file="cv_results.json"
            )

            self._log_model_artifact(searcher.best_estimator_, model)

            print(f"   {search_name} завершён для '{run_name}'")
            print(f"   Лучшие параметры: {best_params}")
            print(f"   Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"   Train MAE: ${train_mae:,.0f}, Test MAE: ${test_mae:,.0f}")

            return searcher.best_estimator_, searcher.best_params_