"""
CatBoost 模型训练脚本 — 为指定站点训练树模型
使用方式:
    python scripts/train_catboost.py [站点代码] [预测小时数]
    python scripts/train_catboost.py 2597A 6
    python scripts/train_catboost.py --all 6      # 训练所有站点
"""
import os
import sys
import time
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import DB_CONFIG, TARGET_SITES, MODEL_DIR

# ─── 命令行参数解析 ──────────────────────────────────────
args = sys.argv[1:]
SITES_TO_TRAIN = []
FORECAST_HOURS = 6

i = 0
while i < len(args):
    if args[i] == "--all":
        SITES_TO_TRAIN = list(TARGET_SITES)
    elif args[i].isdigit() or (args[i].startswith("-") and args[i][1:].isdigit()):
        # 预测小时数
        pass
    else:
        SITES_TO_TRAIN.append(args[i])
    i += 1

# 从参数中提取预测小时数
for a in args:
    a_stripped = a.lstrip("-")
    if a_stripped.isdigit():
        h = int(a_stripped)
        if 1 <= h <= 48:
            FORECAST_HOURS = h

if not SITES_TO_TRAIN:
    SITES_TO_TRAIN = ["2597A"]  # 默认站点

print(f"配置: 站点={SITES_TO_TRAIN}, 预测小时={FORECAST_HOURS}h")

os.makedirs(MODEL_DIR, exist_ok=True)


def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
        f"charset={DB_CONFIG['charset']}"
    )


def prepare_tree_data(df, forecast_hours):
    """为树模型构造特征（完整版）"""
    df_feat = df.copy()

    # 时间特征
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # 滞后特征
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    key_lags = [1, 2, 3, 6, 12, 23]
    for p in pollutants:
        if p in df_feat.columns:
            for lag in key_lags:
                df_feat[f"{p}_lag{lag}"] = df_feat[p].shift(lag)

    # 滚动统计特征
    rolling_windows = [3, 6, 12, 24]
    for p in pollutants:
        if p in df_feat.columns:
            for window in rolling_windows:
                df_feat[f"{p}_mean_{window}h"] = (
                    df_feat[p].rolling(window=window).mean()
                )
                df_feat[f"{p}_std_{window}h"] = (
                    df_feat[p].rolling(window=window).std()
                )

    for window in [3, 6, 12]:
        df_feat[f"AQI_mean_{window}h"] = df_feat["AQI"].rolling(window=window).mean()
        df_feat[f"AQI_std_{window}h"] = df_feat["AQI"].rolling(window=window).std()

    diff_periods = [1, 3, 6, 12]
    for period in diff_periods:
        df_feat[f"AQI_diff_{period}h"] = df_feat["AQI"].diff(period)

    # 目标变量
    target_col = f"AQI_future_{forecast_hours}h"
    df_feat[target_col] = df_feat["AQI"].shift(-forecast_hours)

    # 清理含未来信息的列
    cols_to_drop = [c for c in df_feat.columns if "future" in c.lower() and c != target_col]
    df_feat = df_feat.drop(columns=cols_to_drop, errors="ignore")

    feature_cols = [col for col in df_feat.columns if col != target_col]

    initial_len = len(df_feat)
    df_feat = df_feat.dropna()
    dropped = initial_len - len(df_feat)

    return df_feat[feature_cols], df_feat[target_col], feature_cols, dropped


def evaluate(y_true, y_pred):
    """计算评估指标"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else np.nan
    return r2, rmse, mae, mape


def train_site_model(site_code, forecast_hours, engine):
    """训练单个站点的 CatBoost 模型"""
    print(f"\n{'='*60}")
    print(f"🎯 训练站点: {site_code} | 预测未来 {forecast_hours}h | 模型: CatBoost")
    print(f"{'='*60}")

    # 读取数据
    table_name = f"air_quality_site_{site_code.lower()}"
    try:
        df = pd.read_sql(f"SELECT * FROM `{table_name}` ORDER BY `datetime`", engine)
        print(f"✓ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return None

    if len(df) < 100:
        print(f"✗ 数据不足（{len(df)}条），至少需要100条")
        return None

    # 预处理
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.dropna(subset=["AQI"]).ffill(limit=3).fillna(df.median())

    # 特征工程
    t0 = time.time()
    X, y, feature_names, dropped = prepare_tree_data(df, forecast_hours)
    print(f"✓ 特征: {len(feature_names)}个, 样本: {len(X)}个, 丢弃NaN: {dropped}个")

    # 80/20 切分
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"📊 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        bagging_temperature=1,
        random_state=42,
        verbose=0,
        thread_count=-1,
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    r2, rmse, mae, mape = evaluate(y_test, y_pred)
    train_time = time.time() - t0

    mape_str = f"{mape:.2%}" if not np.isnan(mape) else "N/A"
    print(f"✓ 测试集: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape_str}")
    print(f"⏱ 训练耗时: {train_time:.1f}s")

    # 全量训练最终模型
    print("🔄 使用全部数据训练最终模型...")
    final_model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        bagging_temperature=1,
        random_state=42,
        verbose=0,
        thread_count=-1,
    )
    final_model.fit(X, y)

    # 保存模型
    model_path = os.path.join(
        MODEL_DIR,
        f"aqi_catboost_optimized_model_{site_code}_future{forecast_hours}h.pkl"
    )
    joblib.dump({
        "model": final_model,
        "feature_names": feature_names,
        "site_name": site_code,
        "model_type": "catboost",
        "forecast_hours": forecast_hours,
        "r2_score": r2,
        "rmse": rmse,
    }, model_path)
    print(f"💾 模型已保存: {os.path.basename(model_path)}")

    return {
        "site": site_code,
        "forecast_hours": forecast_hours,
        "r2": r2, "rmse": rmse, "mae": mae, "mape": mape,
        "samples": len(X), "features": len(feature_names),
        "train_time": train_time, "model_path": model_path,
    }


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"🌲 CatBoost 模型训练")
    print(f"   站点: {SITES_TO_TRAIN}")
    print(f"   预测: 未来 {FORECAST_HOURS} 小时")
    print("=" * 60)

    engine = get_engine()
    results = []

    for i, site in enumerate(SITES_TO_TRAIN, 1):
        print(f"\n[{i}/{len(SITES_TO_TRAIN)}]", end="")
        result = train_site_model(site, FORECAST_HOURS, engine)
        if result:
            results.append(result)

    engine.dispose()

    # 汇总
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("📊 训练结果汇总")
        print(f"{'='*60}")
        print(f"{'站点':<8} | {'R²':<8} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<8} | {'耗时(s)':<8}")
        print("-" * 65)
        for _, row in df_results.iterrows():
            mape_s = f"{row['mape']:.2%}" if row['mape'] is not None and not np.isnan(row['mape']) else "N/A"
            print(f"{row['site']:<8} | {row['r2']:<8.4f} | {row['rmse']:<8.2f} | {row['mae']:<8.2f} | {mape_s:<8} | {row['train_time']:<8.1f}")

        if len(results) > 1:
            print("-" * 65)
            avg_data = df_results["mape"].dropna()
            avg_mape = avg_data.mean() if len(avg_data) > 0 else np.nan
            avg_mape_s = f"{avg_mape:.2%}" if not np.isnan(avg_mape) else "N/A"
            print(f"{'平均':<8} | {df_results['r2'].mean():<8.4f} | {df_results['rmse'].mean():<8.2f} | {df_results['mae'].mean():<8.2f} | {avg_mape_s:<8} |")

        # 最佳模型
        best = df_results.sort_values("r2", ascending=False).iloc[0]
        print(f"\n🏆 最佳站点: {best['site']} (R²={best['r2']:.4f})")

        # 保存汇总
        summary_path = os.path.join(MODEL_DIR, f"catboost_training_summary_{FORECAST_HOURS}h.csv")
        df_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"💾 汇总已保存: {summary_path}")

    print("\n✅ 训练完成！")
