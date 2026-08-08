"""
模型对比脚本 — 在指定站点上对比 Prophet/RF/LSTM 性能
使用方式: python scripts/compare_models.py [站点代码] [预测小时数]
"""
import os
import sys
import time
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import DB_CONFIG, COMPARISON_DIR

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch 未安装，LSTM 将跳过")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️ Prophet 未安装，将跳过")

SITE_CODE = sys.argv[1] if len(sys.argv) > 1 else "1916A"
FORECAST_HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

os.makedirs(COMPARISON_DIR, exist_ok=True)
print(f"✅ 输出目录: {COMPARISON_DIR}\n")


def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
        f"charset={DB_CONFIG['charset']}"
    )


def load_raw_data(site_code, engine):
    """加载原始数据"""
    print("=" * 60)
    print(f"加载站点 {site_code} 数据")
    print("=" * 60)

    table_name = f"air_quality_site_{site_code.lower()}"
    try:
        df = pd.read_sql(f"SELECT * FROM `{table_name}` ORDER BY `datetime`", engine)
        print(f"✓ 成功读取 {len(df)} 条")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return None

    if len(df) < 100:
        print(f"✗ 数据量不足（{len(df)}条）")
        return None

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.dropna(subset=["AQI"]).ffill(limit=3).fillna(df.median())
    return df


def prepare_features_unified(df):
    """统一特征工程（树模型用）"""
    df_feat = df.copy()
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    key_lags = [1, 2, 3, 6, 12, 23]
    for p in pollutants:
        if p in df_feat.columns:
            for lag in key_lags:
                df_feat[f"{p}_lag{lag}"] = df_feat[p].shift(lag)
            df_feat[f"{p}_mean_3h"] = df_feat[p].rolling(3).mean()
            df_feat[f"{p}_std_3h"] = df_feat[p].rolling(3).std()

    df_feat["AQI_mean_3h"] = df_feat["AQI"].rolling(3).mean()
    df_feat["AQI_std_3h"] = df_feat["AQI"].rolling(3).std()
    df_feat["AQI_diff_1h"] = df_feat["AQI"].diff(1)

    target_col = f"AQI_future_{FORECAST_HOURS}h"
    df_feat[target_col] = df_feat["AQI"].shift(-FORECAST_HOURS)

    cols_to_drop = [c for c in df_feat.columns if "future" in c.lower() and c != target_col]
    df_feat = df_feat.drop(columns=cols_to_drop, errors="ignore")

    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    print(f"\n✓ 特征准备: {X.shape[1]} 特征, {len(X)} 样本")
    return X, y


def evaluate_tree_cv(model, X, y, model_name):
    """5折时间序列交叉验证"""
    print(f"\n评估: {model_name}")
    print("-" * 50)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {"r2": [], "rmse": [], "mae": []}
    start = time.time()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)

        cv_scores["r2"].append(r2_score(y_val, y_pred))
        cv_scores["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_scores["mae"].append(mean_absolute_error(y_val, y_pred))
        print(f"  Fold {fold+1}/5: R²={cv_scores['r2'][-1]:.4f}, RMSE={cv_scores['rmse'][-1]:.2f}")

    elapsed = time.time() - start
    avg_r2, std_r2 = np.mean(cv_scores["r2"]), np.std(cv_scores["r2"])
    print(f"\n✓ {model_name}: R²={avg_r2:.4f}±{std_r2:.4f}, RMSE={np.mean(cv_scores['rmse']):.2f}, 耗时={elapsed:.1f}s")
    return {"model_name": model_name, "r2_mean": avg_r2, "r2_std": std_r2,
            "rmse": np.mean(cv_scores["rmse"]), "mae": np.mean(cv_scores["mae"]), "time": elapsed}


if __name__ == "__main__":
    print("=" * 60)
    print(f"📊 模型对比实验 — 站点{SITE_CODE}, 未来{FORECAST_HOURS}h")
    print("=" * 60)

    engine = get_engine()
    df_raw = load_raw_data(SITE_CODE, engine)

    if df_raw is None:
        print("❌ 数据加载失败")
        sys.exit(1)

    all_results = []
    X_tree, y_tree = None, None

    # RandomForest
    if X_tree is None:
        X_tree, y_tree = prepare_features_unified(df_raw)
    model = RandomForestRegressor(
        n_estimators=300, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    result = evaluate_tree_cv(model, X_tree, y_tree, "RandomForest")
    result["description"] = "随机森林 (Bagging)"
    all_results.append(result)

    engine.dispose()

    # 汇总
    if all_results:
        df_results = pd.DataFrame(all_results).sort_values("r2_mean", ascending=False)
        print(f"\n\n{'='*60}")
        print("🏆 对比结果")
        print(f"{'='*60}")
        print(f"{'排名':<5} {'模型':<15} {'R²均值':<10} {'R²标准差':<10} {'RMSE':<10} {'耗时(s)':<10}")
        print("-" * 60)
        for idx, row in df_results.iterrows():
            print(f"{list(df_results.index).index(idx)+1:<5} {row['model_name']:<15} {row['r2_mean']:<10.4f} {row['r2_std']:<10.4f} {row['rmse']:<10.2f} {row['time']:<10.1f}")

        csv_path = os.path.join(COMPARISON_DIR, "model_comparison_fair.csv")
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 结果已保存: {csv_path}")

    print("\n✅ 完成！")
