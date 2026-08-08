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
    print("[WARN] PyTorch 未安装，LSTM 将跳过")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("[WARN] Prophet 未安装，将跳过")

SITE_CODE = sys.argv[1] if len(sys.argv) > 1 else "1916A"
FORECAST_HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

os.makedirs(COMPARISON_DIR, exist_ok=True)
print(f"[OK] 输出目录: {COMPARISON_DIR}\n")


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
        print(f"[OK] 成功读取 {len(df)} 条")
    except Exception as e:
        print(f"[FAIL] 读取失败: {e}")
        return None

    if len(df) < 100:
        print(f"[FAIL] 数据量不足（{len(df)}条）")
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

    print(f"\n[OK] 特征准备: {X.shape[1]} 特征, {len(X)} 样本")
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
        print(f"  Fold {fold+1}/5: R2={cv_scores['r2'][-1]:.4f}, RMSE={cv_scores['rmse'][-1]:.2f}")

    elapsed = time.time() - start
    avg_r2, std_r2 = np.mean(cv_scores["r2"]), np.std(cv_scores["r2"])
    print(f"\n[OK] {model_name}: R2={avg_r2:.4f}±{std_r2:.4f}, RMSE={np.mean(cv_scores['rmse']):.2f}, 耗时={elapsed:.1f}s")
    return {"model_name": model_name, "r2_mean": avg_r2, "r2_std": std_r2,
            "rmse": np.mean(cv_scores["rmse"]), "mae": np.mean(cv_scores["mae"]), "time": elapsed}


if __name__ == "__main__":
    print("=" * 60)
    print(f"[START] 模型对比实验 — 站点{SITE_CODE}, 未来{FORECAST_HOURS}h")
    print("=" * 60)

    engine = get_engine()
    df_raw = load_raw_data(SITE_CODE, engine)

    if df_raw is None:
        print("[ERROR] 数据加载失败")
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

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        print(f"\n评估: CatBoost")
        print("-" * 50)
        t0 = time.time()
        cb = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.1,
            l2_leaf_reg=3, random_state=42, verbose=0, thread_count=-1,
        )
        cb_result = evaluate_tree_cv(cb, X_tree, y_tree, "CatBoost")
        cb_result["description"] = "CatBoost (Boosting)"
        all_results.append(cb_result)
    except ImportError:
        print("\n[WARN] CatBoost 未安装，跳过")

    # LSTM
    if HAS_TORCH:
        print(f"\n评估: LSTM")
        print("-" * 50)

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        # LSTM 需要标准化
        scaler_x = MinMaxScaler()
        scaler_y_lstm = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X_tree.values)
        y_scaled = scaler_y_lstm.fit_transform(y_tree.values.reshape(-1, 1)).flatten()

        lstm_scores = {"r2": [], "rmse": [], "mae": []}
        t0 = time.time()
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_tr = torch.tensor(X_scaled[train_idx], dtype=torch.float32).unsqueeze(1)
            X_vl = torch.tensor(X_scaled[val_idx], dtype=torch.float32).unsqueeze(1)
            y_tr = torch.tensor(y_scaled[train_idx], dtype=torch.float32).view(-1, 1)
            y_vl = torch.tensor(y_scaled[val_idx], dtype=torch.float32).view(-1, 1)

            model = LSTMModel(X_tr.shape[-1])
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            for _ in range(50):  # 50 epochs
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(X_tr), y_tr)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_vl).numpy().flatten()
            y_pred = scaler_y_lstm.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler_y_lstm.inverse_transform(y_vl.numpy().reshape(-1, 1)).flatten()

            lstm_scores["r2"].append(r2_score(y_true, y_pred))
            lstm_scores["rmse"].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            lstm_scores["mae"].append(mean_absolute_error(y_true, y_pred))
            print(f"  Fold {fold+1}/5: R2={lstm_scores['r2'][-1]:.4f}, RMSE={lstm_scores['rmse'][-1]:.2f}")

        elapsed = time.time() - t0
        avg_r2, std_r2 = np.mean(lstm_scores["r2"]), np.std(lstm_scores["r2"])
        print(f"\n[OK] LSTM: R2={avg_r2:.4f}±{std_r2:.4f}, RMSE={np.mean(lstm_scores['rmse']):.2f}, 耗时={elapsed:.1f}s")
        all_results.append({"model_name": "LSTM", "r2_mean": avg_r2, "r2_std": std_r2,
                            "rmse": np.mean(lstm_scores["rmse"]),
                            "mae": np.mean(lstm_scores["mae"]), "time": elapsed,
                            "description": "LSTM (深度学习)"})

    # Prophet
    if HAS_PROPHET:
        print(f"\n评估: Prophet")
        print("-" * 50)
        prophet_scores = {"r2": [], "rmse": [], "mae": []}
        t0 = time.time()

        # 为 Prophet CV 准备简单特征数据
        df_p = df_raw.reset_index().rename(columns={"datetime": "ds", "AQI": "y"})
        df_p["y"] = df_p["y"].ffill().fillna(0)
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_p)):
            train_df = df_p.iloc[train_idx][["ds", "y"]]
            val_df = df_p.iloc[val_idx][["ds", "y"]]

            try:
                m = Prophet(yearly_seasonality="auto", weekly_seasonality=True,
                            daily_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(train_df)
                future = m.make_future_dataframe(periods=len(val_df), freq="h")
                forecast = m.predict(future)
                y_pred = forecast.iloc[-len(val_df):]["yhat"].values
                y_true = val_df["y"].values

                prophet_scores["r2"].append(r2_score(y_true, y_pred))
                prophet_scores["rmse"].append(np.sqrt(mean_squared_error(y_true, y_pred)))
                prophet_scores["mae"].append(mean_absolute_error(y_true, y_pred))
                print(f"  Fold {fold+1}/5: R2={prophet_scores['r2'][-1]:.4f}, RMSE={prophet_scores['rmse'][-1]:.2f}")
            except Exception as e:
                print(f"  Fold {fold+1}/5: 失败 ({e})")

        elapsed = time.time() - t0
        if prophet_scores["r2"]:
            avg_r2, std_r2 = np.mean(prophet_scores["r2"]), np.std(prophet_scores["r2"])
            print(f"\n[OK] Prophet: R2={avg_r2:.4f}±{std_r2:.4f}, RMSE={np.mean(prophet_scores['rmse']):.2f}, 耗时={elapsed:.1f}s")
            all_results.append({"model_name": "Prophet", "r2_mean": avg_r2, "r2_std": std_r2,
                                "rmse": np.mean(prophet_scores["rmse"]),
                                "mae": np.mean(prophet_scores["mae"]), "time": elapsed,
                                "description": "Prophet (时序分解)"})
    else:
        print("\n[WARN] Prophet 未安装，跳过对比")

    engine.dispose()

    # 汇总
    if all_results:
        df_results = pd.DataFrame(all_results).sort_values("r2_mean", ascending=False)
        print(f"\n\n{'='*60}")
        print("[RESULT] 对比结果")
        print(f"{'='*60}")
        print(f"{'排名':<5} {'模型':<15} {'R2均值':<10} {'R2标准差':<10} {'RMSE':<10} {'耗时(s)':<10}")
        print("-" * 60)
        for idx, row in df_results.iterrows():
            print(f"{list(df_results.index).index(idx)+1:<5} {row['model_name']:<15} {row['r2_mean']:<10.4f} {row['r2_std']:<10.4f} {row['rmse']:<10.2f} {row['time']:<10.1f}")

        csv_path = os.path.join(COMPARISON_DIR, "model_comparison_fair.csv")
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 结果已保存: {csv_path}")

    print("\n[OK] 完成！")
