import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
import time

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️ Prophet未安装，请运行: py -m pip install prophet")

warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

target_sites = ['2610A', '2611A', '2596A', '2597A', '1916A', '1917A', '3376A', '3377A']

FORECAST_HOURS = 3
LAG_HOURS = 23

output_dir = r'C:\Users\28927\dazuoye\pythonProject3\模型'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 模型保存目录: {output_dir}\n")

engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


class ProphetModelWrapper:
    """Prophet模型包装器，支持序列化"""
    
    def __init__(self, site_name, regressors=None):
        self.site_name = site_name
        self.regressors = regressors or []
        self.model = None
        self.scaler_y = None
    
    def fit(self, df):
        """训练Prophet模型"""
        if not HAS_PROPHET:
            raise ImportError("Prophet未安装")
        
        df_prophet = df.copy()
        
        self.scaler_y = MinMaxScaler()
        target_col = f'AQI_future_{FORECAST_HOURS}h'
        df_prophet['y_scaled'] = self.scaler_y.fit_transform(df_prophet[target_col].values.reshape(-1, 1))
        
        prophet_df = df_prophet[['ds', 'y_scaled'] + self.regressors].copy()
        prophet_df = prophet_df.rename(columns={'y_scaled': 'y'})
        
        self.model = Prophet(
            growth='linear',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        for reg in self.regressors:
            self.model.add_regressor(reg)
        
        self.model.fit(prophet_df)
        return self
    
    def predict(self, df_future):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        prophet_df = df_future[['ds'] + self.regressors].copy()
        forecast = self.model.predict(prophet_df)
        
        predictions_scaled = forecast['yhat'].values
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    def save(self, filepath):
        """保存模型"""
        joblib.dump({
            'site_name': self.site_name,
            'regressors': self.regressors,
            'model': self.model,
            'scaler_y': self.scaler_y
        }, filepath)
    
    @staticmethod
    def load(filepath):
        """加载模型"""
        data = joblib.load(filepath)
        wrapper = ProphetModelWrapper(data['site_name'], data['regressors'])
        wrapper.model = data['model']
        wrapper.scaler_y = data['scaler_y']
        return wrapper


def prepare_prophet_data(df):
    """为Prophet准备数据"""
    df_feat = df.copy()
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    key_lags = [1, 2, 3, 6, 12, 23]
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            for lag in key_lags:
                df_feat[f'{pollutant}_lag{lag}'] = df_feat[pollutant].shift(lag)
    
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            df_feat[f'{pollutant}_mean_3h'] = df_feat[pollutant].rolling(window=3).mean()
            df_feat[f'{pollutant}_std_3h'] = df_feat[pollutant].rolling(window=3).std()
    
    df_feat['AQI_mean_3h'] = df_feat['AQI'].rolling(window=3).mean()
    df_feat['AQI_std_3h'] = df_feat['AQI'].rolling(window=3).std()
    df_feat['AQI_diff_1h'] = df_feat['AQI'].diff(1)
    
    target_col = f'AQI_future_{FORECAST_HOURS}h'
    df_feat[target_col] = df_feat['AQI'].shift(-FORECAST_HOURS)
    
    columns_to_remove = []
    for col in df_feat.columns:
        if 'future' in col.lower() and col != target_col:
            columns_to_remove.append(col)
        if '_lead' in col.lower():
            columns_to_remove.append(col)
    
    if columns_to_remove:
        df_feat = df_feat.drop(columns=columns_to_remove)
    
    df_feat = df_feat.reset_index()
    df_feat = df_feat.rename(columns={'datetime': 'ds', 'AQI': 'y'})
    
    regressor_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO', 
                      'hour', 'day_of_week', 'month', 'is_weekend']
    available_regressors = [col for col in regressor_cols if col in df_feat.columns]
    
    initial_len = len(df_feat)
    df_feat = df_feat.dropna(subset=['ds', 'y'] + available_regressors)
    dropped = initial_len - len(df_feat)
    
    return df_feat, available_regressors, dropped


def train_site_model(site_name):
    """为单个站点训练Prophet模型"""
    print("\n" + "=" * 70)
    print(f"开始训练站点 {site_name} 的Prophet模型")
    print("=" * 70)

    start_time = time.time()

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

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()

    df = df.dropna(subset=['AQI'])
    df = df.ffill(limit=3)
    df = df.fillna(df.median())

    df_prophet, regressors, dropped_count = prepare_prophet_data(df)
    
    print(f"\n✓ 数据准备完成:")
    print(f"  - 样本数量: {len(df_prophet)}")
    print(f"  - 回归变量: {len(regressors)}个")
    print(f"  - 因NaN丢弃: {dropped_count}个")

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {'r2': [], 'rmse': [], 'mae': []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_prophet)):
        df_train = df_prophet.iloc[train_idx].copy()
        df_val = df_prophet.iloc[val_idx].copy()
        
        prophet_wrapper = ProphetModelWrapper(site_name, regressors)
        prophet_wrapper.fit(df_train)
        
        val_pred = prophet_wrapper.predict(df_val)
        y_val_actual = df_val['y'].values
        
        min_len = min(len(val_pred), len(y_val_actual))
        r2 = r2_score(y_val_actual[:min_len], val_pred[:min_len])
        rmse = np.sqrt(mean_squared_error(y_val_actual[:min_len], val_pred[:min_len]))
        mae = mean_absolute_error(y_val_actual[:min_len], val_pred[:min_len])
        
        cv_scores['r2'].append(r2)
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        
        print(f"  Fold {fold+1}/3: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    avg_r2 = np.mean(cv_scores['r2'])
    avg_rmse = np.mean(cv_scores['rmse'])
    avg_mae = np.mean(cv_scores['mae'])
    std_r2 = np.std(cv_scores['r2'])

    print(f"\n✓ 交叉验证结果: R²={avg_r2:.4f}±{std_r2:.4f}, RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}")

    final_model = ProphetModelWrapper(site_name, regressors)
    final_model.fit(df_prophet)

    model_path = os.path.join(output_dir, f'aqi_prophet_model_{site_name}_future{FORECAST_HOURS}h.pkl')
    final_model.save(model_path)

    importance_data = {
        'site': site_name,
        'regressors': ', '.join(regressors),
        'r2': avg_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'samples': len(df_prophet)
    }

    elapsed_time = time.time() - start_time

    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 训练耗时: {elapsed_time:.2f}秒")

    return {
        'site': site_name,
        'r2': avg_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'samples': len(df_prophet),
        'time': elapsed_time,
        'model_path': model_path
    }


if __name__ == '__main__':
    if not HAS_PROPHET:
        print("❌ Prophet未安装，请先安装: py -m pip install prophet")
        exit(1)
    
    print("=" * 70)
    print("开始批量训练所有站点的Prophet AQI预测模型")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  - 预测未来: {FORECAST_HOURS}小时")
    print(f"  - 滞后特征: 前{LAG_HOURS}小时")
    print(f"  - 站点数量: {len(target_sites)}个")

    all_results = []

    for i, site_name in enumerate(target_sites, 1):
        print(f"\n[{i}/{len(target_sites)}] ", end="")
        result = train_site_model(site_name)
        if result is not None:
            all_results.append(result)

    print("\n\n" + "=" * 70)
    print("所有站点模型训练完成！汇总结果:")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n模型性能对比:")
        print("-" * 70)
        print(f"{'站点':<10} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10} | {'样本数':<10} | {'耗时(s)':<8}")
        print("-" * 70)

        for _, row in results_df.iterrows():
            print(
                f"{row['site']:<10} | {row['r2']:<10.4f} | {row['rmse']:<10.2f} | {row['mae']:<10.2f} | {row['samples']:<10} | {row['time']:<8.2f}")

        print("-" * 70)
        print(
            f"{'平均':<10} | {results_df['r2'].mean():<10.4f} | {results_df['rmse'].mean():<10.2f} | {results_df['mae'].mean():<10.2f}")

        summary_path = os.path.join(output_dir, 'prophet_model_training_summary.csv')
        results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 汇总结果已保存: {summary_path}")

        best_model_idx = results_df['r2'].idxmax()
        best_model = results_df.loc[best_model_idx]
        print(f"\n🏆 最佳模型: 站点{best_model['site']}, R²={best_model['r2']:.4f}")
    else:
        print("\n✗ 没有成功训练的模型")

    print("\n✅ 所有任务完成！")
