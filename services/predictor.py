"""
AQI 预测引擎 — 支持 Prophet / CatBoost 双模型，自动选择最优
"""
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from config import (
    DB_CONFIG,
    MODEL_DIR,
    FORECAST_HOURS_LIST,
    PRIMARY_FORECAST_HOURS,
    SITE_DETAILS,
    AQI_LEVELS,
)


# ─── 数据库引擎（模块级复用） ─────────────────────────────
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
            f"charset={DB_CONFIG['charset']}"
        )
    return _engine


# ─── AQI 等级工具函数 ────────────────────────────────────

def get_aqi_info(aqi_value: float) -> dict:
    """根据 AQI 数值获取等级信息"""
    for (min_val, max_val), info in AQI_LEVELS.items():
        if min_val <= aqi_value <= max_val:
            return info
    return AQI_LEVELS[(0, 50)]


# ═══════════════════════════════════════════════════════════
#  Prophet 模型包装器
# ═══════════════════════════════════════════════════════════

class ProphetModelWrapper:
    """Prophet 模型包装器，支持序列化/反序列化"""

    def __init__(self, site_name: str, regressors: list | None = None):
        self.site_name = site_name
        self.regressors = regressors or []
        self.model = None
        self.scaler_y = None
        self.model_type = "prophet"

    def predict(self, df_future: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Prophet 模型未加载")
        prophet_df = df_future[["ds"] + self.regressors].copy()
        forecast = self.model.predict(prophet_df)
        predictions_scaled = forecast["yhat"].values
        return self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

    @staticmethod
    def load(filepath: str) -> "ProphetModelWrapper":
        data = joblib.load(filepath)
        wrapper = ProphetModelWrapper(data["site_name"], data["regressors"])
        wrapper.model = data["model"]
        wrapper.scaler_y = data["scaler_y"]
        return wrapper


# ═══════════════════════════════════════════════════════════
#  CatBoost 模型包装器（树模型通用）
# ═══════════════════════════════════════════════════════════

class CatBoostModelWrapper:
    """CatBoost / XGBoost / LightGBM 等树模型的通用包装器"""

    def __init__(self, site_name: str, feature_names: list | None = None):
        self.site_name = site_name
        self.feature_names = feature_names or []
        self.model = None
        self.model_type = "catboost"

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """使用特征 DataFrame 进行预测"""
        if self.model is None:
            raise ValueError("CatBoost 模型未加载")
        # 确保只使用模型训练时的特征
        available_features = [f for f in self.feature_names if f in df_features.columns]
        missing = set(self.feature_names) - set(available_features)
        if missing:
            # 为缺失特征补 0
            for m in missing:
                df_features[m] = 0
        X = df_features[self.feature_names].values
        return self.model.predict(X)

    @staticmethod
    def load(filepath: str) -> "CatBoostModelWrapper":
        data = joblib.load(filepath)
        wrapper = CatBoostModelWrapper(data.get("site_name", ""), data.get("feature_names", []))
        wrapper.model = data["model"]
        # 兼容不同的 model_type 标记
        wrapper.model_type = data.get("model_type", "catboost").lower()
        return wrapper


# ═══════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════

# 每个站点可配置偏好的模型类型（"prophet" / "catboost" / "auto"）
SITE_MODEL_PREFERENCE = {}

# CatBoost 模型文件名模式（与 demo/2597A尝试.py 输出一致）
CATBOOST_MODEL_PATTERNS = [
    "aqi_catboost_optimized_model_{site}_future{fh}h.pkl",
    "aqi_catboost_model_{site}_future{fh}h.pkl",
    "aqi_randomforest_optimized_model_{site}_future{fh}h.pkl",
    "aqi_xgboost_default_model_{site}_future{fh}h.pkl",
    "aqi_lightgbm_default_model_{site}_future{fh}h.pkl",
]


def load_models() -> dict:
    """加载所有站点的预测模型（支持 Prophet + CatBoost 双引擎）

    优先级：
    1. 如果站点在 SITE_MODEL_PREFERENCE 中指定了模型类型 → 优先加载该类型
    2. 如果 CatBoost 模型存在 → 使用 CatBoost
    3. 否则 → 使用 Prophet（默认）

    Returns:
        dict: {站点代码: {预测小时数: 模型实例}}
    """
    models = {}
    for site in SITE_DETAILS:
        models[site] = {}
        preference = SITE_MODEL_PREFERENCE.get(site, "prophet")

        for fh in FORECAST_HOURS_LIST:
            loaded = False

            # ── 尝试加载 CatBoost 模型 ──
            if preference in ("catboost", "auto"):
                for pattern in CATBOOST_MODEL_PATTERNS:
                    cb_path = os.path.join(MODEL_DIR, pattern.format(site=site, fh=fh))
                    if os.path.exists(cb_path):
                        try:
                            wrapper = CatBoostModelWrapper.load(cb_path)
                            models[site][fh] = wrapper
                            print(f"[OK] 已加载站点 {site}（{SITE_DETAILS[site]['name']}）"
                                  f"的{fh}h {wrapper.model_type.upper()} 模型")
                            loaded = True
                        except Exception as e:
                            print(f"[WARN] 站点 {site} 的{fh}h CatBoost 模型加载失败: {e}")
                        break

            # ── 回退到 Prophet 模型 ──
            if not loaded:
                prophet_path = os.path.join(
                    MODEL_DIR, f"aqi_prophet_model_{site}_future{fh}h.pkl"
                )
                if os.path.exists(prophet_path):
                    try:
                        models[site][fh] = ProphetModelWrapper.load(prophet_path)
                        print(f"[OK] 已加载站点 {site}（{SITE_DETAILS[site]['name']}）"
                              f"的{fh}h Prophet 模型")
                        loaded = True
                    except Exception as e:
                        print(f"[FAIL] 站点 {site} 的{fh}h Prophet 模型加载失败: {e}")

            if not loaded:
                print(f"[FAIL] 未找到站点 {site} 的{fh}h模型文件")

    return models


# ═══════════════════════════════════════════════════════════
#  特征工程
# ═══════════════════════════════════════════════════════════

def _build_prophet_features(df: pd.DataFrame) -> pd.DataFrame:
    """为 Prophet 模型构造特征（预测时用）"""
    df_feat = df.copy()
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    key_lags = [1, 2, 3, 6, 12, 23]
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            for lag in key_lags:
                if len(df_feat) >= lag:
                    df_feat[f"{pollutant}_lag{lag}"] = df_feat[pollutant].shift(lag)
            for window in [3]:
                if len(df_feat) >= window:
                    df_feat[f"{pollutant}_mean_{window}h"] = (
                        df_feat[pollutant].iloc[-window:].mean()
                    )
                    df_feat[f"{pollutant}_std_{window}h"] = (
                        df_feat[pollutant].iloc[-window:].std()
                    )

    if len(df_feat) >= 3:
        df_feat["AQI_mean_3h"] = df_feat["AQI"].iloc[-3:].mean()
        df_feat["AQI_std_3h"] = df_feat["AQI"].iloc[-3:].std()
    if len(df_feat) >= 2:
        df_feat["AQI_diff_1h"] = df_feat["AQI"].iloc[-1] - df_feat["AQI"].iloc[-2]

    df_feat = df_feat.reset_index()
    df_feat = df_feat.rename(columns={"datetime": "ds", "AQI": "y"})
    return df_feat


def _build_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """为树模型（CatBoost/RF/XGBoost）构造特征

    这是 demo/2597A尝试.py 中 prepare_tree_data() 的完整版本，
    包含丰富的滞后特征和滚动统计特征。
    """
    df_feat = df.copy()
    df_feat["hour"] = df_feat.index.hour
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
    key_lags = [1, 2, 3, 6, 12, 23]
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            for lag in key_lags:
                df_feat[f"{pollutant}_lag{lag}"] = df_feat[pollutant].shift(lag)

    rolling_windows = [3, 6, 12, 24]
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            for window in rolling_windows:
                df_feat[f"{pollutant}_mean_{window}h"] = (
                    df_feat[pollutant].rolling(window=window).mean()
                )
                df_feat[f"{pollutant}_std_{window}h"] = (
                    df_feat[pollutant].rolling(window=window).std()
                )

    for window in [3, 6, 12]:
        df_feat[f"AQI_mean_{window}h"] = df_feat["AQI"].rolling(window=window).mean()
        df_feat[f"AQI_std_{window}h"] = df_feat["AQI"].rolling(window=window).std()

    diff_periods = [1, 3, 6, 12]
    for period in diff_periods:
        df_feat[f"AQI_diff_{period}h"] = df_feat["AQI"].diff(period)

    # 保留 datetime 列用于时间偏移
    df_feat = df_feat.reset_index()
    return df_feat


# ═══════════════════════════════════════════════════════════
#  预测
# ═══════════════════════════════════════════════════════════

def predict_aqi(
    site_name: str, models: dict, force_fetch: bool = False
) -> dict | None:
    """对指定站点执行多时间步 AQI 预测

    支持 Prophet 和 CatBoost 两种模型，自动根据模型类型选择特征工程策略。

    Args:
        site_name: 站点代码（如 "1916A"）
        models: 已加载的模型字典
        force_fetch: 是否强制实时获取数据

    Returns:
        dict: {"multi_preds": {fh: {"aqi": float, "time": datetime}}, "current_time": datetime}
    """
    if site_name not in models:
        print(f"[WARN] 站点 {site_name} 的模型未加载")
        return None

    location = SITE_DETAILS[site_name]["location"]
    table_name = f"air_quality_site_{site_name.lower()}"
    query = f"SELECT * FROM `{table_name}` ORDER BY `datetime` DESC LIMIT 100"

    # 数据新鲜度阈值（小时）：超过此值将触发爬虫更新
    DATA_FRESHNESS_THRESHOLD_HOURS = 2

    try:
        df = pd.read_sql(query, _get_engine())

        # ── 判断是否需要触发爬虫 ──
        # 条件1：数据量不足
        need_fetch = len(df) < 30
        if need_fetch:
            print(f"[WARN] 站点 {site_name} 数据库数据不足（{len(df)}条）")

        # 条件2：数据过时（即使数据量够，也可能很久没更新）
        if not need_fetch and len(df) > 0:
            df["datetime"] = pd.to_datetime(df["datetime"])
            latest_in_db = df["datetime"].max()
            hours_behind = (pd.Timestamp.now() - latest_in_db).total_seconds() / 3600
            if hours_behind > DATA_FRESHNESS_THRESHOLD_HOURS:
                need_fetch = True
                print(
                    f"[WARN] 站点 {site_name} 数据过时 "
                    f"（最新: {latest_in_db.strftime('%m-%d %H:%M')}, "
                    f"距今 {hours_behind:.0f}h），触发实时更新..."
                )
            else:
                print(
                    f"  站点 {site_name} 数据新鲜 "
                    f"（{latest_in_db.strftime('%m-%d %H:%M')}, {hours_behind:.1f}h前）"
                )

        # ── 触发爬虫获取实时数据 ──
        if need_fetch:
            try:
                from services.data_fetcher import get_or_fetch_data

                get_or_fetch_data(site_name, location, force_fetch=True)
                # 重新读取数据库
                df = pd.read_sql(query, _get_engine())
            except Exception as e:
                print(f"[WARN] 实时获取失败 ({e})，使用现有数据继续")

        # ── 再次检查数据是否可用 ──
        if len(df) < 30:
            print(f"[FAIL] 数据仍然不足（{len(df)}条），无法预测")
            return None

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        latest_time = df.index[-1]

        # 诊断信息
        data_start = df.index[0]
        data_end = df.index[-1]
        print(f"  数据时间范围: {data_start.strftime('%Y-%m-%d %H:%M')} ~ "
              f"{data_end.strftime('%Y-%m-%d %H:%M')}")
        hours_behind = (pd.Timestamp.now() - data_end).total_seconds() / 3600
        if hours_behind > 24:
            print(f"  [WARN] 最新数据距现在 {hours_behind:.0f} 小时，非实时数据")

        # 准备特征（两种类型都准备，根据模型类型选用）
        df_prophet = _build_prophet_features(df)
        df_tree = _build_tree_features(df)

        # 多时间步预测
        multi_preds = {}
        for fh in FORECAST_HOURS_LIST:
            if fh not in models[site_name]:
                print(f"[WARN] 站点 {site_name} 的{fh}h模型未加载，跳过")
                continue

            model = models[site_name][fh]
            future_ds = latest_time + timedelta(hours=fh)

            try:
                # ── 根据模型类型选择预测路径 ──
                if isinstance(model, CatBoostModelWrapper):
                    # CatBoost / 树模型预测
                    future_row = df_tree.iloc[[-1]].copy()
                    predicted_aqi = model.predict(future_row)[0]
                else:
                    # Prophet 模型预测
                    future_row = df_prophet.iloc[[-1]].copy()
                    future_row["ds"] = future_ds

                    # 填充缺失的回归变量
                    for col in model.regressors:
                        if col in future_row.columns and future_row[col].isna().any():
                            historical_median = df_prophet[col].median()
                            future_row[col] = future_row[col].fillna(historical_median)

                    missing = [
                        c for c in model.regressors
                        if c in future_row.columns and future_row[c].isna().any()
                    ]
                    if missing:
                        print(f"[WARN] {fh}h 预测仍有缺失值: {missing}，使用0填充")
                        for col in missing:
                            future_row[col] = future_row[col].fillna(0)

                    predicted_aqi = model.predict(future_row)[0]

                # 约束 AQI 范围 [0, 500]
                predicted_aqi = max(0, min(500, predicted_aqi))

                multi_preds[fh] = {
                    "aqi": round(float(predicted_aqi), 1),
                    "time": future_ds,
                }

            except Exception as e:
                print(f"[ERROR] {fh}h 预测出错 ({type(model).__name__}): {e}")
                continue

        if not multi_preds:
            return None

        return {"multi_preds": multi_preds, "current_time": latest_time}

    except Exception as e:
        print(f"[ERROR] 预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_location_aqi(location: str, models: dict) -> dict | None:
    """综合预测某个地区所有站点的 AQI（多时间跨度）"""
    location_sites = [
        site for site, info in SITE_DETAILS.items() if info["location"] == location
    ]
    if not location_sites:
        print(f"[WARN] 未找到地区 {location} 的站点")
        return None

    print(f"[LOC] 地区 {location} 共有 {len(location_sites)} 个监测站点")

    predictions = []
    for site in location_sites:
        print(f"  正在预测站点 {site}（{SITE_DETAILS[site]['name']}）...")
        pred = predict_aqi(site, models)
        if pred is not None:
            pred["site_code"] = site
            pred["site_name"] = SITE_DETAILS[site]["name"]
            predictions.append(pred)

    if not predictions:
        print(f"[FAIL] 地区 {location} 所有站点预测失败")
        return None

    latest_time = max(p["current_time"] for p in predictions)
    return {
        "site_count": len(predictions),
        "current_time": latest_time,
        "sites": predictions,
    }
