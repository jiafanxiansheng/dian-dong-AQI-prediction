import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ 警告: PyTorch未安装，LSTM模型将不可用")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️ 警告: Prophet未安装，Prophet模型将不可用")

warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

FORECAST_HOURS = 3
LAG_HOURS = 23
SITE_CODE = '1916A'

output_dir = r'C:\Users\28927\dazuoye\pythonProject3\模型对比'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 输出目录: {output_dir}\n")

engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


def load_raw_data(site_code):
    """加载原始数据（不做特征工程）"""
    print("=" * 70)
    print(f"加载站点 {site_code} 的原始数据")
    print("=" * 70)

    table_name = f'air_quality_site_{site_code.lower()}'

    try:
        query = f"SELECT * FROM `{table_name}` ORDER BY `datetime`"
        df = pd.read_sql(query, engine)
        print(f"✓ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        return None

    if len(df) < 100:
        print(f"✗ 数据量不足（{len(df)}条）")
        return None

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()
    df = df.dropna(subset=['AQI'])
    df = df.ffill(limit=3)
    df = df.fillna(df.median())
    
    return df


def prepare_features_unified(df):
    """
    为树模型准备统一的基础特征集
    保持简洁，避免过度特征工程导致不公平对比
    """
    df_feat = df.copy()
    
    # 1. 时间特征
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. 周期性特征
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    # 3. 关键滞后特征（只保留核心时间点）
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    key_lags = [1, 2, 3, 6, 12, 23]
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            for lag in key_lags:
                df_feat[f'{pollutant}_lag{lag}'] = df_feat[pollutant].shift(lag)
    
    # 4. 短期滚动统计（只保留3h和6h）
    for pollutant in pollutants:
        if pollutant in df_feat.columns:
            df_feat[f'{pollutant}_mean_3h'] = df_feat[pollutant].rolling(window=3).mean()
            df_feat[f'{pollutant}_std_3h'] = df_feat[pollutant].rolling(window=3).std()
    
    # 5. AQI自身的关键特征
    df_feat['AQI_mean_3h'] = df_feat['AQI'].rolling(window=3).mean()
    df_feat['AQI_std_3h'] = df_feat['AQI'].rolling(window=3).std()
    df_feat['AQI_diff_1h'] = df_feat['AQI'].diff(1)
    
    # 创建目标变量
    target_col = f'AQI_future_{FORECAST_HOURS}h'
    df_feat[target_col] = df_feat['AQI'].shift(-FORECAST_HOURS)
    
    # 排除未来信息
    columns_to_remove = []
    for col in df_feat.columns:
        if 'future' in col.lower() and col != target_col:
            columns_to_remove.append(col)
        if '_lead' in col.lower():
            columns_to_remove.append(col)
    
    if columns_to_remove:
        df_feat = df_feat.drop(columns=columns_to_remove)
    
    features_to_exclude = [target_col]
    X = df_feat.drop(columns=[c for c in features_to_exclude if c in df_feat.columns])
    y = df_feat[target_col]
    
    # 删除缺失值
    initial_len = len(X)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    dropped_samples = initial_len - len(X)
    
    print(f"\n✓ 树模型特征准备完成:")
    print(f"  - 特征数量: {X.shape[1]}")
    print(f"  - 样本数量: {len(X)}")
    print(f"  - 因NaN丢弃样本: {dropped_samples}个 ({dropped_samples/initial_len*100:.1f}%)")
    
    return X, y, df_feat


def prepare_lstm_sequences(df, sequence_length=24):
    """
    为LSTM准备序列数据
    使用原始数据 + 简单时间编码
    """
    df_seq = df.copy()
    
    # 基础特征（不包含复杂的衍生特征）
    base_features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO', 'AQI']
    available_features = [f for f in base_features if f in df_seq.columns]
    
    # 添加时间特征
    df_seq['hour_sin'] = np.sin(2 * np.pi * df_seq.index.hour / 24)
    df_seq['hour_cos'] = np.cos(2 * np.pi * df_seq.index.hour / 24)
    available_features += ['hour_sin', 'hour_cos']
    
    # 创建目标变量
    target_col = f'AQI_future_{FORECAST_HOURS}h'
    df_seq[target_col] = df_seq['AQI'].shift(-FORECAST_HOURS)
    
    # 删除缺失值
    df_seq = df_seq.dropna()
    
    # 提取特征和目标
    feature_data = df_seq[available_features].values
    target_data = df_seq[target_col].values
    
    # 标准化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    feature_scaled = scaler_X.fit_transform(feature_data)
    target_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()
    
    # 构建序列
    X_sequences, y_sequences = [], []
    for i in range(len(feature_scaled) - sequence_length):
        X_sequences.append(feature_scaled[i:i+sequence_length])
        y_sequences.append(target_scaled[i+sequence_length])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    print(f"\n✓ LSTM序列数据准备完成:")
    print(f"  - 序列长度: {sequence_length}")
    print(f"  - 特征维度: {len(available_features)}")
    print(f"  - 样本数量: {len(X_seq)}")
    print(f"  - 使用的特征: {available_features}")
    
    return X_seq, y_seq, scaler_X, scaler_y, sequence_length, available_features


def prepare_prophet_data(df):
    """
    为Prophet准备数据格式
    """
    df_prophet = df.copy()
    
    # 重置索引，使datetime成为列
    df_prophet = df_prophet.reset_index()
    
    # 创建目标变量
    target_col = f'AQI_future_{FORECAST_HOURS}h'
    df_prophet[target_col] = df_prophet['AQI'].shift(-FORECAST_HOURS)
    
    # 删除包含未来信息的列
    cols_to_drop = [col for col in df_prophet.columns if 'future' in col.lower() and col != target_col]
    df_prophet = df_prophet.drop(columns=cols_to_drop)
    
    # 重命名列为Prophet格式
    df_prophet = df_prophet.rename(columns={'datetime': 'ds', 'AQI': 'y'})
    
    # 选择要作为回归变量的特征
    regressor_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    available_regressors = [col for col in regressor_cols if col in df_prophet.columns]
    
    # 删除缺失值
    df_prophet = df_prophet.dropna(subset=['ds', 'y'] + available_regressors)
    
    print(f"\n✓ Prophet数据准备完成:")
    print(f"  - 样本数量: {len(df_prophet)}")
    print(f"  - 回归变量: {available_regressors}")
    
    return df_prophet, available_regressors


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


def train_and_predict_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                           epochs=30, batch_size=64, learning_rate=0.001):
    """训练LSTM并返回预测结果"""
    if not HAS_TORCH:
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train_seq.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).cpu().numpy()
        val_pred = model(X_val_tensor).cpu().numpy()
    
    return train_pred, val_pred


class ProphetWrapper:
    """Prophet模型包装器，兼容sklearn接口"""
    
    def __init__(self, growth='linear', yearly_seasonality=True, 
                 weekly_seasonality=True, daily_seasonality=True,
                 changepoint_prior_scale=0.05):
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.scaler_y = None
        self.regressors = []
    
    def get_params(self, deep=True):
        """sklearn兼容的get_params方法"""
        return {
            'growth': self.growth,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale
        }
    
    def set_params(self, **params):
        """sklearn兼容的set_params方法"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, df_prophet, regressors):
        if not HAS_PROPHET:
            raise ImportError("Prophet未安装")
        
        self.regressors = regressors
        
        self.scaler_y = MinMaxScaler()
        df_prophet['y_scaled'] = self.scaler_y.fit_transform(df_prophet['y'].values.reshape(-1, 1))
        
        prophet_df = df_prophet[['ds', 'y_scaled'] + regressors].copy()
        prophet_df = prophet_df.rename(columns={'y_scaled': 'y'})
        
        self.model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        for reg in regressors:
            self.model.add_regressor(reg)
        
        self.model.fit(prophet_df)
        return self
    
    def predict(self, df_prophet):
        if not HAS_PROPHET or self.model is None:
            raise ValueError("模型未训练或Prophet未安装")
        
        prophet_df = df_prophet[['ds'] + self.regressors].copy()
        forecast = self.model.predict(prophet_df)
        
        predictions_scaled = forecast['yhat'].values
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions


def evaluate_tree_models_cv(model, X, y, model_name):
    """评估树模型（使用时间序列交叉验证）"""
    print(f"\n评估模型: {model_name}")
    print("-" * 70)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
    start_time = time.time()
    
    best_fold_model = None
    best_fold_r2 = -np.inf

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
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
        
        if r2 > best_fold_r2:
            best_fold_r2 = r2
            best_fold_model = fold_model

        print(f"  Fold {fold+1}/5: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    elapsed_time = time.time() - start_time
    avg_r2 = np.mean(cv_scores['r2'])
    avg_rmse = np.mean(cv_scores['rmse'])
    avg_mae = np.mean(cv_scores['mae'])
    std_r2 = np.std(cv_scores['r2'])

    print(f"\n✓ {model_name} 交叉验证结果:")
    print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  RMSE = {avg_rmse:.2f}")
    print(f"  MAE = {avg_mae:.2f}")
    print(f"  耗时 = {elapsed_time:.2f}秒")
    
    try:
        if hasattr(best_fold_model, 'feature_importances_'):
            importances = best_fold_model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 10 重要特征:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']:30s}: {row['importance']:.4f}")
    except Exception as e:
        print(f"  ⚠ 无法提取特征重要性: {e}")

    return {
        'model_name': model_name,
        'r2_mean': avg_r2,
        'r2_std': std_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'time': elapsed_time
    }


def evaluate_lstm_model(df, site_code):
    """评估LSTM模型"""
    if not HAS_TORCH:
        print("\n⚠️ 跳过LSTM评估（PyTorch未安装）")
        return None
    
    print(f"\n评估模型: LSTM")
    print("-" * 70)
    
    try:
        X_seq, y_seq, scaler_X, scaler_y, seq_len, features = prepare_lstm_sequences(df, sequence_length=24)
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            X_train_seq, X_val_seq = X_seq[train_idx], X_seq[val_idx]
            y_train_seq, y_val_seq = y_seq[train_idx], y_seq[val_idx]
            
            train_pred, val_pred = train_and_predict_lstm(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                epochs=30, batch_size=64, learning_rate=0.001
            )
            
            if train_pred is None:
                continue
            
            # 反标准化预测结果
            val_pred_actual = scaler_y.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            y_val_actual = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_actual, val_pred_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_actual, val_pred_actual)
            r2 = r2_score(y_val_actual, val_pred_actual)
            
            cv_scores['mse'].append(mse)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            
            print(f"  Fold {fold+1}/5: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        elapsed_time = time.time() - start_time
        
        if len(cv_scores['r2']) == 0:
            print("  ✗ 所有fold均失败")
            return None
        
        avg_r2 = np.mean(cv_scores['r2'])
        avg_rmse = np.mean(cv_scores['rmse'])
        avg_mae = np.mean(cv_scores['mae'])
        std_r2 = np.std(cv_scores['r2'])
        
        print(f"\n✓ LSTM 交叉验证结果:")
        print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
        print(f"  RMSE = {avg_rmse:.2f}")
        print(f"  MAE = {avg_mae:.2f}")
        print(f"  耗时 = {elapsed_time:.2f}秒")
        
        return {
            'model_name': 'LSTM',
            'r2_mean': avg_r2,
            'r2_std': std_r2,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'time': elapsed_time
        }
    
    except Exception as e:
        print(f"  ✗ LSTM评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_prophet_model(df, site_code):
    """评估Prophet模型"""
    if not HAS_PROPHET:
        return None
    
    print(f"\n评估模型: Prophet")
    print("-" * 70)
    
    df_prophet, regressors = prepare_prophet_data(df)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {'r2': [], 'rmse': [], 'mae': []}
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_prophet)):
        df_train = df_prophet.iloc[train_idx].copy()
        df_val = df_prophet.iloc[val_idx].copy()
        
        prophet_model = ProphetWrapper()
        prophet_model.fit(df_train, regressors)
        
        val_pred = prophet_model.predict(df_val)
        y_val_actual = df_val['y'].values
        
        min_len = min(len(val_pred), len(y_val_actual))
        r2 = r2_score(y_val_actual[:min_len], val_pred[:min_len])
        rmse = np.sqrt(mean_squared_error(y_val_actual[:min_len], val_pred[:min_len]))
        mae = mean_absolute_error(y_val_actual[:min_len], val_pred[:min_len])
        
        cv_scores['r2'].append(r2)
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        
        print(f"  Fold {fold+1}/5: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    
    elapsed_time = time.time() - start_time
    
    if len(cv_scores['r2']) == 0:
        return None
    
    avg_r2 = np.mean(cv_scores['r2'])
    std_r2 = np.std(cv_scores['r2'])
    avg_rmse = np.mean(cv_scores['rmse'])
    avg_mae = np.mean(cv_scores['mae'])
    
    print(f"\n✓ Prophet结果: R²={avg_r2:.4f}±{std_r2:.4f}, RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}, 耗时={elapsed_time:.1f}秒")
    
    return {
        'model_name': 'Prophet',
        'r2_mean': avg_r2,
        'r2_std': std_r2,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'time': elapsed_time
    }

if __name__ == '__main__':
    print("=" * 70)
    print(" 模型对比实验 - AQI预测（公平对比版）")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  - 测试站点: {SITE_CODE}")
    print(f"  - 预测未来: {FORECAST_HOURS}小时")
    print(f"  - 滞后特征: 前{LAG_HOURS}小时")
    print(f"  - 交叉验证: 5折时间序列")
    print(f"  - PyTorch可用: {HAS_TORCH}")
    print(f"  - Prophet可用: {HAS_PROPHET}")

    df_raw = load_raw_data(SITE_CODE)
    if df_raw is None:
        print("\n❌ 数据加载失败，程序退出")
        exit(1)

    all_results = []
    
    # ==================== 评估树模型 ====================
    X_tree, y_tree, df_tree = prepare_features_unified(df_raw)
    
    tree_models = {
        'CatBoost_Optimized': CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.1,
            l2_leaf_reg=3, bagging_temperature=1,
            random_state=42, verbose=0, thread_count=-1
        ),
        'RF_Optimized': RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_split=10, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
    }
    
    tree_descriptions = {
        'CatBoost_Optimized': 'CatBoost-优化参数 (Boosting家族代表)',
        'RF_Optimized': '随机森林-优化参数 (Bagging家族代表)',
    }

    print(f"\n{'='*70}")
    print("第一阶段：树模型对比")
    print(f"{'='*70}")
    
    for model_name, model in tree_models.items():
        result = evaluate_tree_models_cv(model, X_tree, y_tree, model_name)
        if result:
            result['description'] = tree_descriptions[model_name]
            all_results.append(result)
            
            model_path = os.path.join(output_dir, f'{model_name.lower()}_model.pkl')
            joblib.dump(model, model_path)
            print(f"  ✓ 模型已保存: {model_path}\n")

    # ==================== 评估LSTM ====================
    print(f"\n{'='*70}")
    print("第二阶段：LSTM评估")
    print(f"{'='*70}")
    
    lstm_result = evaluate_lstm_model(df_raw, SITE_CODE)
    if lstm_result:
        lstm_result['description'] = '长短期记忆网络 (深度学习时序模型)'
        all_results.append(lstm_result)

    # ==================== 评估Prophet ====================
    print(f"\n{'='*70}")
    print("第三阶段：Prophet评估")
    print(f"{'='*70}")
    
    prophet_result = evaluate_prophet_model(df_raw, SITE_CODE)
    if prophet_result:
        prophet_result['description'] = 'Facebook Prophet (统计时序模型)'
        all_results.append(prophet_result)

    # ==================== 汇总结果 ====================
    if len(all_results) == 0:
        print("\n❌ 没有模型成功评估")
        exit(1)

    print("\n\n" + "=" * 70)
    print("最终模型对比结果汇总")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
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

    summary_path = os.path.join(output_dir, 'model_comparison_fair.csv')
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比结果已保存: {summary_path}")



