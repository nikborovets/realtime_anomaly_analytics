import itertools
import logging
import ast
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ts_toolkit.metrics import global_metrics
from src.models.decomposition_model_for_avg import DecompositionImprovedTrendModel

from pathlib import Path
import datetime

from ts_toolkit.split import three_way_split
from ts_toolkit.viz import plot_history_forecast

# --- базовый логгер ---
LOGS_ROOT = Path("tune_catboost")
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Добавляем FileHandler ко всему процессу (один на весь запуск)
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("tuning.log") for h in root_logger.handlers):
    fh_root = logging.FileHandler(LOGS_ROOT / "tuning.log")
    fh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(fh_root)

# Убедимся, что вывод идёт и в консоль
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    sh_root = logging.StreamHandler()
    sh_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(sh_root)


def build_param_grid(fast: bool = False) -> List[Dict]:
    """Возвращает список словарей с комбинациями гиперпараметров.

    Args:
        fast: Если True — минимальный grid для быстрого теста.
    """

    if fast:
        param_grid = {
            "trend_model_type": ["global"],
            "trend_window_size": [960],
            "depth": [10],
            "l2_leaf_reg": [10],
            "learning_rate": [0.1],
            "iterations": [500],
            "rsm": [0.8],
            "early_stopping_rounds": [100],
        }
        # Разворачиваем grid в список словарей для fast-режима
        keys, values = zip(*param_grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        # КРАЕВЫЕ 6 КОМБИНАЦИЙ (минимум времени на запуск)
        combos = []

        # 1-2: global с «мягкой» и «жёсткой» регуляризацией
        combos.append({
            "trend_model_type": "global",
            "trend_window_size": 960,
            "depth": 10,
            "l2_leaf_reg": 20,
            "learning_rate": 0.03,
            "iterations": 3000,
            "rsm": 0.8,
            "early_stopping_rounds": 500,
            "use_cyclic_features": True,
            "ema_alpha": 0.2,
        })
        combos.append({
            "trend_model_type": "global",
            "trend_window_size": 960,
            "depth": 15,
            "l2_leaf_reg": 40,
            "learning_rate": 0.015,
            "iterations": 3000,
            "rsm": 0.75,
            "early_stopping_rounds": 300,
            "use_cyclic_features": False,
            "ema_alpha": 0.1,
        })

        # 3-6: local, окно 960 и 5760, по две крайние конфигурации
        for win in [960, 5760]:
            combos.append({
                "trend_model_type": "local",
                "trend_window_size": win,
                "depth": 10,
                "l2_leaf_reg": 20,
                "learning_rate": 0.03,
                "iterations": 3000,
                "rsm": 0.8,
                "early_stopping_rounds": 300,
                "use_cyclic_features": True,
                "ema_alpha": 0.2,
            })
            combos.append({
                "trend_model_type": "local",
                "trend_window_size": win,
                "depth": 15,
                "l2_leaf_reg": 40,
                "learning_rate": 0.015,
                "iterations": 3000,
                "rsm": 0.75,
                "early_stopping_rounds": 300,
                "use_cyclic_features": False,
                "ema_alpha": 0.1,
            })

    return combos


def _param_dict_to_name(params: Dict) -> str:
    """Создает читаемое имя модели на основе словаря параметров."""
    parts = [f"{k}={v}" for k, v in params.items() if k not in {"trend_window_size"}]
    safe = [p.replace(".", "p") for p in parts]
    return "__".join(safe)


def evaluate_params(
    params: Dict,
    df_full: pd.DataFrame,
    target_col: str,
    combo_id: int,
    n_splits: int = 5,
    test_size: int = 900,
) -> Dict:
    """Возвращает средние метрики для конкретного набора параметров."""

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    metrics_list = []

    model_name = f"comb_{combo_id}_{_param_dict_to_name(params)}"

    # --- подготовка директорий и логгеров ---
    plots_dir = Path(f"tune_catboost/plots/{model_name}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    # Добавляем FileHandler один раз
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(plots_dir / "results.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    logger.info("Начинаю оценку комбинации %s", model_name)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_full)):
        df_train_fold = df_full.iloc[train_idx]
        df_val_fold = df_full.iloc[val_idx]

        cb_params = {
            "depth": params["depth"],
            "l2_leaf_reg": params["l2_leaf_reg"],
            "learning_rate": params["learning_rate"],
            "iterations": params["iterations"],
            "rsm": params["rsm"],
            "early_stopping_rounds": params["early_stopping_rounds"],
            "use_cyclic_features": params["use_cyclic_features"],
            "ema_alpha": params["ema_alpha"],
        }

        model = DecompositionImprovedTrendModel(
            horizon=len(df_val_fold),
            trend_model_type=params["trend_model_type"],
            trend_window_size=params["trend_window_size"],
            **cb_params,
        )

        model.fit(
            train_df=df_train_fold,
            target_col=target_col,
            val_df=df_val_fold,
        )

        y_pred = model.predict(df_hist=df_train_fold)
        y_true = df_val_fold[target_col].reindex(y_pred.index).dropna()
        y_pred = y_pred.reindex(y_true.index).dropna()

        if len(y_true) == 0:
            logger.warning("Fold %d: нет пересечения данных, пропускаю", fold_idx + 1)
            continue

        fold_metrics = global_metrics(y_true, y_pred)
        metrics_list.append(fold_metrics)
        logger.info("Fold %d metrics: %s", fold_idx + 1, fold_metrics)

        # --- сохранение графика ---
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        hist_start = y_pred.index[0] - pd.Timedelta(minutes=30)
        plot_history_forecast(
            history=df_train_fold.loc[hist_start:y_pred.index[0], target_col],
            forecast=y_pred,
            actual=y_true,
            title=f"Blind forecast vs actual — fold {fold_idx + 1}",
            filename=str(plots_dir / f"cv_fold_{fold_idx + 1}_forecast_{ts}.png"),
        )

    if not metrics_list:
        # Возвращаем огромные значения, чтобы комбинация точно не победила
        return {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}

    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    logger.info("Average CV metrics: %s", avg_metrics)

    
    df_train_val, df_train_val_val, df_hold = three_way_split(df_full, train_ratio=0.8, val_ratio=0.19)
    # --- Обучаем финальную модель на всех данных и сохраняем ---
    final_model = DecompositionImprovedTrendModel(
        horizon=len(df_hold),
        trend_model_type=params["trend_model_type"],
        trend_window_size=params["trend_window_size"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        learning_rate=params["learning_rate"],
        iterations=params["iterations"],
        rsm=params["rsm"],
        early_stopping_rounds=params["early_stopping_rounds"],
        use_cyclic_features=params["use_cyclic_features"],
        ema_alpha=params["ema_alpha"],
    )
    final_model.fit(df_train_val, target_col=target_col, val_df=df_train_val_val)
    y_pred = final_model.predict(df_hist=pd.concat([df_train_val, df_train_val_val]))
    y_true = df_hold[target_col].reindex(y_pred.index).dropna()
    y_pred = y_pred.reindex(y_true.index).dropna()

    if not y_pred.empty and not y_true.empty:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hold_hist_start = y_pred.index[0] - pd.Timedelta(minutes=30)
        plot_history_forecast(
            history=df_train_val.loc[hold_hist_start:y_pred.index[0], target_col],
            forecast=y_pred,
            actual=y_true,
            title="Blind forecast vs actual — hold-out",
            filename=str(plots_dir / f"hold_out_forecast_{ts}.png"),
        )
    else:
        logger.warning("Hold-out: нет пересечения данных, график не сохранён")

    models_dir = Path(f"tune_catboost/models/{model_name}")
    final_model.save(str(models_dir))
    logger.info("Final model saved to %s", models_dir)
    logger.info("Final model CatBoost params: %s", final_model.cb_params)

    return avg_metrics


def _load_evaluated_param_sets(log_path: Path = LOGS_ROOT / "tuning.log") -> set:
    """Считывает tuning.log и возвращает set из каноничных tuple(sorted(dict.items()))."""

    evaluated = set()
    if not log_path.exists():
        return evaluated

    for line in log_path.read_text().splitlines():
        if "Параметры:" in line:
            try:
                dict_str = line.split("Параметры:")[1].strip()
                params = ast.literal_eval(dict_str)
                evaluated.add(tuple(sorted(params.items())))
            except Exception:
                continue
    return evaluated


def tune_model(
    df_full: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    test_size: int = 900,
    fast_grid: bool = False,
):
    all_params = build_param_grid(fast=fast_grid)
    # --- фильтруем уже протестированное ---
    evaluated = _load_evaluated_param_sets()
    params_to_try = [p for p in all_params if tuple(sorted(p.items())) not in evaluated]

    if not params_to_try:
        logging.info("Все комбинации уже были оценены. Нечего делать — выхожу.")
        return pd.DataFrame()

    logging.info("Всего комбинаций: %d (из них новых %d)", len(all_params), len(params_to_try))

    results = []
    for idx, params in enumerate(params_to_try, 1):
        new_inx = idx + 20
        logging.info("==== Комбинация %d/%d ====" , new_inx, len(params_to_try)+20)
        logging.info("Параметры: %s", params)
        metrics = evaluate_params(params, df_full, target_col, new_inx, n_splits, test_size)
        logging.info("Средние метрики: %s", metrics)
        results.append({**params, **metrics})

    results_df = pd.DataFrame(results)
    results_df.sort_values("RMSE", inplace=True)
    logging.info("\nТоп-5 лучших комбинаций по RMSE:\n%s", results_df.head())
    return results_df


if __name__ == "__main__":
    import argparse
    from src.data_loader import fetch_frame
    from ts_toolkit.io import clean_timeseries

    parser = argparse.ArgumentParser(description="Hyper-parameter tuning for DecompositionImprovedTrendModel")
    parser.add_argument("--start-date", type=str, default="2024-11-25 18:00:00")
    parser.add_argument("--end-date", type=str, default="2024-12-11 12:10:00")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-filename", type=str, default="common_cad_avg1h_20241125_20241211.parquet")
    parser.add_argument("--target-col", type=str, default="common_cad_avg1h")
    parser.add_argument("--fast-grid", action="store_true", help="Использовать минимальный набор параметров для быстрого теста")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=900)
    args = parser.parse_args()

    df_raw = fetch_frame(
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=args.use_cache,
        cache_filename=args.cache_filename,
    )

    # Попытка найти колонку с длинным именем и переименовать
    long_cols = [c for c in df_raw.columns if c.startswith(args.target_col)]
    if long_cols:
        df_raw.rename(columns={long_cols[0]: args.target_col}, inplace=True)

    df_clean = clean_timeseries(df_raw, args.target_col)

    tune_model(
        df_full=df_clean,
        target_col=args.target_col,
        n_splits=args.n_splits,
        test_size=args.test_size,
        fast_grid=args.fast_grid,
    ) 