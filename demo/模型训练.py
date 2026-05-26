import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ==================== 数据库配置 ====================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

# 目标站点列表
target_sites = ['2610A', '2611A', '2596A', '2597A', '1916A', '1917A', '3376A', '3377A']

# ==================== 配置参数 ====================
FORECAST_HOURS = 3  # 预测未来3小时
USE_TIME_SERIES_CV = True  # 是否使用时间序列交叉验证
LAG_HOURS = 23  # 使用前23小时的滞后特征（加上当前时刻共24个时间点）

# ==================== 创建模型保存目录 ====================
output_dir = r'C:\Users\28927\dazuoye\pythonProject3\模型'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 模型保存目录: {output_dir}\n")

# ==================== 数据库连接 ====================
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


# ==================== 定义模型训练函数 ====================
def train_site_model(site_name):
    """为单个站点训练模型"""
    print("\n" + "=" * 70)
    print(f"开始训练站点 {site_name} 的模型")
    print("=" * 70)

    start_time = time.time()

    # 读取数据
    table_name = f'air_quality_site_{site_name.lower()}'

    try:
        query = f"SELECT * FROM `{table_name}` ORDER BY `datetime`"
        df = pd.read_sql(query, engine)
        print(f"✓ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        return None

    if len(df) < 100:
        print(f"✗ 数据量不足（{len(df)}条），跳过该站点")
        return None

    # 数据预处理
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()

    df = df.dropna(subset=['AQI'])
    df = df.fillna(df.median())

    # 特征工程
    # 1. 时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2. 周期性特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 3. 滞后特征（前1-23小时，加上当前时刻共24个时间点）
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    for pollutant in pollutants:
        if pollutant in df.columns:
            for lag in range(1, LAG_HOURS + 1):
                df[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)

    # 4. 滚动窗口特征
    rolling_windows = [3, 6, 12, 24]
    for pollutant in pollutants:
        if pollutant in df.columns:
            for window in rolling_windows:
                df[f'{pollutant}_mean_{window}h'] = df[pollutant].rolling(window=window).mean()
                df[f'{pollutant}_std_{window}h'] = df[pollutant].rolling(window=window).std()
                df[f'{pollutant}_min_{window}h'] = df[pollutant].rolling(window=window).min()
                df[f'{pollutant}_max_{window}h'] = df[pollutant].rolling(window=window).max()

    # 5. 变化率特征
    for pollutant in pollutants:
        if pollutant in df.columns:
            df[f'{pollutant}_diff_1h'] = df[pollutant].diff(1)
            df[f'{pollutant}_diff_3h'] = df[pollutant].diff(3)
            df[f'{pollutant}_diff_6h'] = df[pollutant].diff(6)

    # 创建目标变量
    df[f'AQI_future_{FORECAST_HOURS}h'] = df['AQI'].shift(-FORECAST_HOURS)

    # 删除NaN值
    df = df.dropna()

    # 确保没有数据泄漏
    columns_to_remove = []
    for col in df.columns:
        if 'future' in col.lower() and col != f'AQI_future_{FORECAST_HOURS}h':
            columns_to_remove.append(col)
        if '_lead' in col.lower():
            columns_to_remove.append(col)

    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)

    features_to_exclude = ['AQI', f'AQI_future_{FORECAST_HOURS}h']
    target_col = f'AQI_future_{FORECAST_HOURS}h'

    X = df.drop(columns=[c for c in features_to_exclude if c in df.columns])
    y = df[target_col]

    print(f"✓ 特征数量: {X.shape[1]}, 样本数量: {len(X)}")

    # 模型训练与验证
    if USE_TIME_SERIES_CV:
        tscv = TimeSeriesSplit(n_splits=5)

        cv_scores = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }

        final_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            fold_model.fit(X_train_fold, y_train_fold)

            y_val_pred = fold_model.predict(X_val_fold)

            mse = mean_squared_error(y_val_fold, y_val_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_fold, y_val_pred)
            r2 = r2_score(y_val_fold, y_val_pred)

            cv_scores['mse'].append(mse)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)

        avg_r2 = np.mean(cv_scores['r2'])
        avg_rmse = np.mean(cv_scores['rmse'])
        avg_mae = np.mean(cv_scores['mae'])

        print(f"✓ 交叉验证结果: R²={avg_r2:.4f}, RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}")

        # 使用全部数据训练最终模型
        final_model.fit(X, y)

    else:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        final_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)

        avg_r2 = r2_score(y_test, y_pred)
        avg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        avg_mae = mean_absolute_error(y_test, y_pred)

        print(f"✓ 测试结果: R²={avg_r2:.4f}, RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}")

    # 计算特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 保存模型
    model_path = os.path.join(output_dir, f'aqi_rf_model_{site_name}_future{FORECAST_HOURS}h_lag{LAG_HOURS}.pkl')
    joblib.dump(final_model, model_path)

    # 保存特征重要性
    importance_path = os.path.join(output_dir, f'feature_importance_{site_name}.csv')
    feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')

    elapsed_time = time.time() - start_time

    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 特征重要性已保存: {importance_path}")
    print(f"✓ 训练耗时: {elapsed_time:.2f}秒")

    return {
        'site': site_name,
        'r2': avg_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'features': X.shape[1],
        'samples': len(X),
        'time': elapsed_time,
        'model_path': model_path
    }


# ==================== 主程序：循环训练所有站点 ====================
print("=" * 70)
print("开始批量训练所有站点的AQI预测模型")
print("=" * 70)
print(f"\n配置参数:")
print(f"  - 预测未来: {FORECAST_HOURS}小时")
print(f"  - 滞后特征: 前{LAG_HOURS}小时（当前时刻 + 过去23小时 = 24个时间点）")
print(f"  - 交叉验证: {'启用' if USE_TIME_SERIES_CV else '禁用'}")
print(f"  - 站点数量: {len(target_sites)}个")

# 存储所有站点的训练结果
all_results = []

# 循环训练每个站点
for i, site_name in enumerate(target_sites, 1):
    print(f"\n[{i}/{len(target_sites)}] ", end="")
    result = train_site_model(site_name)
    if result is not None:
        all_results.append(result)

# ==================== 汇总结果 ====================
print("\n\n" + "=" * 70)
print("所有站点模型训练完成！汇总结果:")
print("=" * 70)

if all_results:
    results_df = pd.DataFrame(all_results)

    print("\n模型性能对比:")
    print("-" * 70)
    print(f"{'站点':<10} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10} | {'特征数':<8} | {'样本数':<10} | {'耗时(s)':<8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(
            f"{row['site']:<10} | {row['r2']:<10.4f} | {row['rmse']:<10.2f} | {row['mae']:<10.2f} | {row['features']:<8} | {row['samples']:<10} | {row['time']:<8.2f}")

    print("-" * 70)
    print(
        f"{'平均':<10} | {results_df['r2'].mean():<10.4f} | {results_df['rmse'].mean():<10.2f} | {results_df['mae'].mean():<10.2f}")

    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'model_training_summary.csv')
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 汇总结果已保存: {summary_path}")

    # 找出最佳模型
    best_model_idx = results_df['r2'].idxmax()
    best_model = results_df.loc[best_model_idx]
    print(f"\n🏆 最佳模型: 站点{best_model['site']}, R²={best_model['r2']:.4f}")
else:
    print("\n✗ 没有成功训练的模型")

print("\n✅ 所有任务完成！")
