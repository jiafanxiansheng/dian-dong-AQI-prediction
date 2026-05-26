import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# ==================== 配置参数 ====================
FORECAST_HOURS = 3  # 预测未来3小时
LAG_HOURS = 23  # 使用前23小时的滞后特征
SITE_CODE = '1916A'  # 测试站点

# ==================== 创建输出目录 ====================
output_dir = r'C:\Users\28927\dazuoye\pythonProject3\模型对比'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 输出目录: {output_dir}\n")

# ==================== 数据库连接 ====================
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


# ==================== 数据加载与特征工程 ====================
def load_and_prepare_data(site_code):
    """加载数据并进行特征工程"""
    print("=" * 70)
    print(f"加载并处理站点 {site_code} 的数据")
    print("=" * 70)
    
    table_name = f'air_quality_site_{site_code.lower()}'
    
    try:
        query = f"SELECT * FROM `{table_name}` ORDER BY `datetime`"
        df = pd.read_sql(query, engine)
        print(f"✓ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        return None, None
    
    if len(df) < 100:
        print(f"✗ 数据量不足（{len(df)}条）")
        return None, None
    
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
    
    # 3. 滞后特征（前1-23小时）
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
    
    # 排除未来信息
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
    
    return X, y


# ==================== 树模型配置 ====================
def get_tree_models():
    """定义所有树模型及其参数配置"""
    
    models = {
        # ==================== Random Forest ====================
        'RF_Default': RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'RF_Optimized': RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'RF_HighDepth': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'RF_LowDepth': RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        # ==================== Extra Trees ====================
        'ET_Default': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'ET_Optimized': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'ET_Fast': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=42,
            n_jobs=-1
        ),
        
        # ==================== Gradient Boosting ====================
        'GB_Default': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        
        'GB_Optimized': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ),
        
        'GB_Slow': GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=42
        ),
        
        'GB_Fast': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.2,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.7,
            random_state=42
        ),
        
        # ==================== XGBoost ====================
        'XGB_Default': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        'XGB_Optimized': XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        'XGB_Deep': XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0.2,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        'XGB_Fast': XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        # ==================== LightGBM ====================
        'LGB_Default': LGBMRegressor(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'LGB_Optimized': LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'LGB_Leaf': LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'LGB_Fast': LGBMRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        # ==================== CatBoost ====================
        'CatBoost_Default': CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0,
            thread_count=-1
        ),
        
        'CatBoost_Optimized': CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            bagging_temperature=1,
            random_state=42,
            verbose=0,
            thread_count=-1
        ),
        
        'CatBoost_Slow': CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=5,
            random_state=42,
            verbose=0,
            thread_count=-1
        ),
    }
    
    return models


# ==================== 模型说明 ====================
def get_model_descriptions():
    """获取每个模型的参数说明"""
    descriptions = {
        'RF_Default': '随机森林-默认参数 (100棵树)',
        'RF_Optimized': '随机森林-优化参数 (300棵树, min_samples_split=10)',
        'RF_HighDepth': '随机森林-高深度 (max_depth=20)',
        'RF_LowDepth': '随机森林-低深度防过拟合 (max_depth=10, 强正则化)',
        
        'ET_Default': '极端随机树-默认参数',
        'ET_Optimized': '极端随机树-优化参数',
        'ET_Fast': '极端随机树-快速版本 (限制深度和特征)',
        
        'GB_Default': '梯度提升树-默认参数',
        'GB_Optimized': '梯度提升树-优化参数 (subsample=0.8)',
        'GB_Slow': '梯度提升树-慢速高精度 (500棵树, lr=0.05)',
        'GB_Fast': '梯度提升树-快速版本 (100棵树, lr=0.2)',
        
        'XGB_Default': 'XGBoost-默认参数',
        'XGB_Optimized': 'XGBoost-优化参数 (gamma正则化)',
        'XGB_Deep': 'XGBoost-深树版本 (max_depth=10)',
        'XGB_Fast': 'XGBoost-快速版本',
        
        'LGB_Default': 'LightGBM-默认参数',
        'LGB_Optimized': 'LightGBM-优化参数 (L1/L2正则化)',
        'LGB_Leaf': 'LightGBM-Leaf-wise生长策略',
        'LGB_Fast': 'LightGBM-快速版本',
        
        'CatBoost_Default': 'CatBoost-默认参数',
        'CatBoost_Optimized': 'CatBoost-优化参数 (bagging)',
        'CatBoost_Slow': 'CatBoost-慢速高精度 (500轮)',
    }
    return descriptions


# ==================== 时间序列交叉验证 ====================
def evaluate_model_cv(model, X, y, model_name):
    """使用时间序列交叉验证评估模型"""
    print(f"\n评估模型: {model_name}")
    print("-" * 70)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # 预测
        y_val_pred = fold_model.predict(X_val_fold)
        
        # 计算指标
        mse = mean_squared_error(y_val_fold, y_val_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        r2 = r2_score(y_val_fold, y_val_pred)
        
        cv_scores['mse'].append(mse)
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        
        print(f"  Fold {fold+1}/5: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # 计算平均指标
    avg_r2 = np.mean(cv_scores['r2'])
    avg_rmse = np.mean(cv_scores['rmse'])
    avg_mae = np.mean(cv_scores['mae'])
    std_r2 = np.std(cv_scores['r2'])
    
    print(f"\n✓ {model_name} 交叉验证结果:")
    print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  RMSE = {avg_rmse:.2f}")
    print(f"  MAE = {avg_mae:.2f}")
    print(f"  耗时 = {elapsed_time:.2f}秒")
    
    return {
        'model_name': model_name,
        'r2_mean': avg_r2,
        'r2_std': std_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'time': elapsed_time
    }


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("🌲 树模型对比实验 - AQI预测")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  - 测试站点: {SITE_CODE}")
    print(f"  - 预测未来: {FORECAST_HOURS}小时")
    print(f"  - 滞后特征: 前{LAG_HOURS}小时")
    print(f"  - 交叉验证: 5折时间序列")
    
    # 加载数据
    X, y = load_and_prepare_data(SITE_CODE)
    
    if X is None:
        print("\n❌ 数据加载失败，程序退出")
        exit(1)
    
    # 获取树模型
    models = get_tree_models()
    descriptions = get_model_descriptions()
    
    print(f"\n 将要评估 {len(models)} 个树模型:")
    for i, (name, desc) in enumerate(descriptions.items(), 1):
        print(f"  {i:2d}. {name:<20s} - {desc}")
    
    # 评估所有模型
    all_results = []
    
    for model_name, model in models.items():
        result = evaluate_model_cv(model, X, y, model_name)
        result['description'] = descriptions.get(model_name, '')
        all_results.append(result)
        
        # 保存模型
        model_path = os.path.join(output_dir, f'{model_name.lower()}_model.pkl')
        joblib.dump(model, model_path)
        print(f"  ✓ 模型已保存: {model_path}\n")
    
    # ==================== 汇总结果 ====================
    print("\n\n" + "=" * 70)
    print("📊 树模型对比结果汇总")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    
    # 按R²排序
    results_df = results_df.sort_values('r2_mean', ascending=False).reset_index(drop=True)
    
    print("\n模型性能排名:")
    print("-" * 100)
    print(f"{'排名':<6} | {'模型':<22} | {'R² (均值)':<12} | {'R² (标准差)':<12} | {'RMSE':<10} | {'MAE':<10} | {'耗时(s)':<10}")
    print("-" * 100)
    
    for idx, row in results_df.iterrows():
        print(
            f"{idx+1:<6} | {row['model_name']:<22} | {row['r2_mean']:<12.4f} | {row['r2_std']:<12.4f} | "
            f"{row['rmse']:<10.2f} | {row['mae']:<10.2f} | {row['time']:<10.2f}"
        )
    
    print("-" * 100)
    
    # 保存结果
    summary_path = os.path.join(output_dir, 'tree_model_comparison_results.csv')
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比结果已保存: {summary_path}")
    
    # 找出最佳模型
    best_model = results_df.iloc[0]
    print(f"\n🏆 最佳模型: {best_model['model_name']}")
    print(f"   说明: {best_model['description']}")
    print(f"   R² = {best_model['r2_mean']:.4f} ± {best_model['r2_std']:.4f}")
    print(f"   RMSE = {best_model['rmse']:.2f}")
    print(f"   MAE = {best_model['mae']:.2f}")
    
    # 按模型家族分类分析
    print("\n" + "=" * 70)
    print(" 按模型家族分类对比")
    print("=" * 70)
    
    model_families = {
        'Random Forest': [name for name in results_df['model_name'] if name.startswith('RF_')],
        'Extra Trees': [name for name in results_df['model_name'] if name.startswith('ET_')],
        'Gradient Boosting': [name for name in results_df['model_name'] if name.startswith('GB_')],
        'XGBoost': [name for name in results_df['model_name'] if name.startswith('XGB_')],
        'LightGBM': [name for name in results_df['model_name'] if name.startswith('LGB_')],
        'CatBoost': [name for name in results_df['model_name'] if name.startswith('CatBoost_')]
    }
    
    for family, model_names in model_families.items():
        family_results = results_df[results_df['model_name'].isin(model_names)]
        if len(family_results) > 0:
            best_in_family = family_results.iloc[0]
            avg_r2 = family_results['r2_mean'].mean()
            print(f"\n {family}:")
            print(f"   最佳: {best_in_family['model_name']} (R²={best_in_family['r2_mean']:.4f})")
            print(f"   平均: R²={avg_r2:.4f}")
            print(f"   变体数: {len(family_results)}")
    
    # Top 5 推荐
    print("\n" + "=" * 70)
    print("🎯 Top 5 推荐模型")
    print("=" * 70)
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(f"\n  #{idx+1}: {row['model_name']}")
        print(f"      {row['description']}")
        print(f"      R² = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
    
    print("\n✅ 树模型对比完成！")
