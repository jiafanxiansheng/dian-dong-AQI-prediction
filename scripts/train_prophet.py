"""
Prophet 模型训练脚本 — 为所有站点训练 AQI 预测模型
使用方式: python scripts/train_prophet.py [预测小时数]
"""
import os
import sys
import time
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("❌ Prophet 未安装，请运行: pip install prophet")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import DB_CONFIG, TARGET_SITES, MODEL_DIR

FORECAST_HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LAG_HOURS = 23

os.makedirs(MODEL_DIR, exist_ok=True)


def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
        f"charset={DB_CONFIG['charset']}"
    )


class ProphetModelWrapper:
    """Prophet 模型包装器"""

    def __init__(self, site_name, regressors=None):
        self.site_name = site_name
        self.regressors = regressors or []
        self.model = None
        self.scaler_y = None

    def fit(self, df):
        df_prophet = df.copy()
        self.scaler_y = MinMaxScaler()
        target_col = f"AQI_future_{FORECAST_HOURS}h"
        df_prophet["y_scaled"] = self.scaler_y.fit_transform(
            df_prophet[target_col].values.reshape(-1, 1)
        )

        prophet_df = df_prophet[["ds", "y_scaled"] + self.regressors].copy()
        prophet_df = prophet_df.rename(columns={"y_scaled": "y"})

        try:
            from prophet.make_holidays import make_holidays_df
            years = list(range(df_prophet["ds"].min().year, df_prophet["ds"].max().year + 2))
            holidays = make_holidays_df(year_list=years, country="CN")
            self.model = Prophet(
                holidays=holidays,
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=True, changepoint_prior_scale=0.05,
                seasonality_prior_scale=5.0, holidays_prior_scale=5.0,
            )
        except Exception:
            self.model = Prophet(
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=True, changepoint_prior_scale=0.05,
            )

        for reg in self.regressors:
            self.model.add_regressor(reg)
        self.model.fit(prophet_df)
        return self

    def predict(self, df_future):
        prophet_df = df_future[["ds"] + self.regressors].copy()
        forecast = self.model.predict(prophet_df)
        predictions_scaled = forecast["yhat"].values
        return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    def save(self, filepath):
        joblib.dump({
            "site_name": self.site_name, "regressors": self.regressors,
            "model": self.model, "scaler_y": self.scaler_y,
        }, filepath)

    @staticmethod
    def load(filepath):
        data = joblib.load(filepath)
        wrapper = ProphetModelWrapper(data["site_name"], data["regressors"])
        wrapper.model = data["model"]
        wrapper.scaler_y = data["scaler_y"]
        return wrapper


def prepare_prophet_data(df):
    """特征工程"""
    df_feat = df.copy()
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    key_lags = [1, 2, 3, 6, 12, 23]
    for p in pollutants:
        if p in df_feat.columns:
            for lag in key_lags:
                df_feat[f"{p}_lag{lag}"] = df_feat[p].shift(lag)

    for p in pollutants:
        if p in df_feat.columns:
            for w in [3, 6, 12, 24]:
                df_feat[f"{p}_mean_{w}h"] = df_feat[p].rolling(window=w).mean()
                df_feat[f"{p}_std_{w}h"] = df_feat[p].rolling(window=w).std()

    for w in [3, 6, 12]:
        df_feat[f"AQI_mean_{w}h"] = df_feat["AQI"].rolling(window=w).mean()
        df_feat[f"AQI_std_{w}h"] = df_feat["AQI"].rolling(window=w).std()

    for period in [1, 3, 6, 12]:
        df_feat[f"AQI_diff_{period}h"] = df_feat["AQI"].diff(period)

    target_col = f"AQI_future_{FORECAST_HOURS}h"
    df_feat[target_col] = df_feat["AQI"].shift(-FORECAST_HOURS)

    # 删除含未来信息的列
    cols_to_drop = [c for c in df_feat.columns if "future" in c.lower() and c != target_col]
    df_feat = df_feat.drop(columns=cols_to_drop, errors="ignore")

    df_feat = df_feat.reset_index().rename(columns={"datetime": "ds", "AQI": "y"})

    regressor_cols = [
        "PM2.5", "PM10", "SO2", "NO2", "O3", "CO",
        "hour", "day_of_week", "month", "is_weekend",
    ]
    available_regressors = [c for c in regressor_cols if c in df_feat.columns]

    initial_len = len(df_feat)
    df_feat = df_feat.dropna(subset=["ds", "y"] + available_regressors)
    dropped = initial_len - len(df_feat)

    return df_feat, available_regressors, dropped


def train_site_model(site_name, engine):
    """训练单个站点的 Prophet 模型"""
    print(f"\n{'='*60}")
    print(f"训练站点 {site_name}（预测未来{FORECAST_HOURS}h）")
    print(f"{'='*60}")
    start_time = time.time()

    table_name = f"air_quality_site_{site_name.lower()}"
    try:
        df = pd.read_sql(f"SELECT * FROM `{table_name}` ORDER BY `datetime`", engine)
        print(f"✓ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return None

    if len(df) < 100:
        print(f"✗ 数据不足（{len(df)}条）")
        return None

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.dropna(subset=["AQI"]).ffill(limit=3).fillna(df.median())

    df_prophet, regressors, dropped = prepare_prophet_data(df)
    if df_prophet is None:
        return None

    print(f"✓ 数据准备完成: {len(df_prophet)} 样本, {len(regressors)} 个回归变量, 丢弃 {dropped}")

    # Hold-out 80/20
    train_size = int(len(df_prophet) * 0.8)
    df_train = df_prophet.iloc[:train_size]
    df_test = df_prophet.iloc[train_size:]

    print(f"📊 训练集: {len(df_train)}, 测试集: {len(df_test)}")

    model = ProphetModelWrapper(site_name, regressors)
    model.fit(df_train)

    # 评估
    test_pred = model.predict(df_test)
    y_true = df_test["y"].values
    min_len = min(len(test_pred), len(y_true))
    r2 = r2_score(y_true[:min_len], test_pred[:min_len])
    rmse = np.sqrt(mean_squared_error(y_true[:min_len], test_pred[:min_len]))
    mae = mean_absolute_error(y_true[:min_len], test_pred[:min_len])

    mask = y_true[:min_len] > 0
    mape = mean_absolute_percentage_error(y_true[:min_len][mask], test_pred[:min_len][mask]) if mask.sum() > 0 else np.nan

    mape_str = f"{mape:.2%}" if not np.isnan(mape) else "N/A"
    print(f"✓ 测试集: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape_str}")

    # 全量训练最终模型
    print("  使用全部数据训练最终模型...")
    final_model = ProphetModelWrapper(site_name, regressors)
    final_model.fit(df_prophet)

    model_path = os.path.join(MODEL_DIR, f"aqi_prophet_model_{site_name}_future{FORECAST_HOURS}h.pkl")
    final_model.save(model_path)

    elapsed = time.time() - start_time
    print(f"✓ 模型已保存: {os.path.basename(model_path)} (耗时 {elapsed:.1f}s)")

    return {
        "site": site_name, "r2": r2, "rmse": rmse, "mae": mae,
        "mape": mape if not np.isnan(mape) else None,
        "samples": len(df_prophet), "time": elapsed, "model_path": model_path,
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 Prophet 模型批量训练（未来{FORECAST_HOURS}小时）")
    print("=" * 60)
    print(f"\n配置: {len(TARGET_SITES)} 个站点, Hold-out 80/20, 节假日效应启用")

    engine = get_engine()
    results = []

    for i, site in enumerate(TARGET_SITES, 1):
        print(f"\n[{i}/{len(TARGET_SITES)}] ", end="")
        result = train_site_model(site, engine)
        if result:
            results.append(result)

    engine.dispose()

    # 汇总
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n\n{'='*60}")
        print("📊 训练结果汇总")
        print(f"{'='*60}")
        print(f"{'站点':<8} | {'R²':<8} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8}")
        print("-" * 50)
        for _, row in df_results.iterrows():
            mape_s = f"{row['mape']:.2%}" if row['mape'] is not None else "N/A"
            print(f"{row['site']:<8} | {row['r2']:<8.4f} | {row['rmse']:<8.2f} | {row['mae']:<8.2f} | {mape_s:<8}")
        print("-" * 50)
        avg_mape = df_results["mape"].dropna().mean()
        print(f"{'平均':<8} | {df_results['r2'].mean():<8.4f} | {df_results['rmse'].mean():<8.2f} | {df_results['mae'].mean():<8.2f} | {avg_mape:.2%}")

        summary_path = os.path.join(MODEL_DIR, f"training_summary_{FORECAST_HOURS}h.csv")
        df_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 汇总已保存: {summary_path}")

    print("\n✅ 所有任务完成！")
