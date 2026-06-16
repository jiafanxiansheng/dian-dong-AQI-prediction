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

# 新增：时序模型相关库
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

    # ✅ 修复pandas警告：使用ffill()替代fillna(method='ffill')
    df = df.ffill(limit=3)  # 最多向前填充3小时
    df = df.fillna(df.median())  # 剩余的用中位数填充
    
    # 特征工程
    # 1. 时间特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # ✅ 新增：时间窗口特征（捕捉早晚高峰）
    df['is_morning_rush'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)  # 早高峰
    df['is_evening_rush'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)  # 晚高峰
    df['is_night'] = df['hour'].apply(lambda x: 1 if 22 <= x or x <= 5 else 0)  # 夜间
    
    # 2. 周期性特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ✅ 新增：AQI的滚动统计（关键！）
    df['AQI_mean_3h'] = df['AQI'].rolling(window=3).mean()
    df['AQI_std_3h'] = df['AQI'].rolling(window=3).std()
    df['AQI_mean_6h'] = df['AQI'].rolling(window=6).mean()
    df['AQI_std_6h'] = df['AQI'].rolling(window=6).std()
    df['AQI_mean_12h'] = df['AQI'].rolling(window=12).mean()
    df['AQI_trend_3h'] = df['AQI'].diff(3)  # 3小时变化趋势
    
    # 3. 滞后特征（✅ 修复：使用完整的24小时滞后特征）
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
    for pollutant in pollutants:
        if pollutant in df.columns:
            for lag in range(1, LAG_HOURS + 1):  # 使用完整的1-24小时滞后
                df[f'{pollutant}_lag{lag}'] = df[pollutant].shift(lag)
    
    # 4. 滚动窗口特征（✅ 优化：增加24小时窗口）
    rolling_windows = [3, 6, 12, 24]  # 新增24小时窗口
    for pollutant in pollutants:
        if pollutant in df.columns:
            for window in rolling_windows:
                df[f'{pollutant}_mean_{window}h'] = df[pollutant].rolling(window=window).mean()
                df[f'{pollutant}_std_{window}h'] = df[pollutant].rolling(window=window).std()
    
    # 5. 变化率特征（✅ 优化：增加中长期变化率）
    diff_periods = [1, 3, 6, 12, 24]  # 新增6h、12h、24h变化率
    for pollutant in pollutants:
        if pollutant in df.columns:
            for period in diff_periods:
                df[f'{pollutant}_diff_{period}h'] = df[pollutant].diff(period)

    # 创建目标变量
    df[f'AQI_future_{FORECAST_HOURS}h'] = df['AQI'].shift(-FORECAST_HOURS)
    
    # ✅ 关键修复：先排除未来信息，再删除NaN
    columns_to_remove = []
    for col in df.columns:
        if 'future' in col.lower() and col != f'AQI_future_{FORECAST_HOURS}h':
            columns_to_remove.append(col)
        if '_lead' in col.lower():
            columns_to_remove.append(col)
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
    
    # ✅ 关键修复：保留当前AQI作为特征！只排除目标变量
    features_to_exclude = [f'AQI_future_{FORECAST_HOURS}h']  # 只排除目标，保留当前AQI
    target_col = f'AQI_future_{FORECAST_HOURS}h'
    
    X = df.drop(columns=[c for c in features_to_exclude if c in df.columns])
    y = df[target_col]
    
    # ✅ 调试：检查当前AQI是否在特征中
    has_aqi_feature = 'AQI' in X.columns
    print(f"\n🔍 调试信息:")
    print(f"  - 当前AQI是否在特征中: {has_aqi_feature}")
    if has_aqi_feature:
        print(f"  - AQI列名: 'AQI'")
        print(f"  - AQI数据范围: [{X['AQI'].min():.1f}, {X['AQI'].max():.1f}]")
        print(f"  - AQI与目标变量的相关性: {X['AQI'].corr(y):.4f}")
    else:
        print(f"  ⚠️  警告：当前AQI不在特征中！")
        aqi_cols = [col for col in X.columns if 'aqi' in col.lower()]
        print(f"  - 包含AQI的列: {aqi_cols}")
    
    # ✅ 改进的缺失值处理：最后统一删除NaN
    initial_len = len(X)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    dropped_samples = initial_len - len(X)
    
    # ✅ 重新计算特征数量
    time_features_count = sum(1 for col in ['hour', 'day_of_week', 'month', 'is_weekend', 
                                              'is_morning_rush', 'is_evening_rush', 'is_night'] 
                               if col in X.columns)
    periodic_features_count = sum(1 for col in ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] 
                                   if col in X.columns)
    aqi_rolling_count = sum(1 for col in X.columns if col.startswith('AQI_'))
    lag_count = sum(1 for col in X.columns if '_lag' in col)
    rolling_count = sum(1 for col in X.columns if '_mean_' in col or '_std_' in col)
    diff_count = sum(1 for col in X.columns if '_diff_' in col)
    
    print(f"\n✓ 特征数量: {X.shape[1]}, 样本数量: {len(X)}")
    print(f"  - 时间特征: {time_features_count}个")
    print(f"  - 周期性特征: {periodic_features_count}个")
    print(f"  - AQI滚动统计: {aqi_rolling_count}个 (✅ 新增)")
    print(f"  - 滞后特征: {lag_count}个 (✅ 完整24小时)")
    print(f"  - 滚动窗口: {rolling_count}个 (✅ 含24h窗口)")
    print(f"  - 变化率特征: {diff_count}个 (✅ 含中长期)")
    if has_aqi_feature:
        print(f"  - 当前AQI: 1个 (✅ 关键特征)")
    print(f"  - 总计: {X.shape[1]}个")
    print(f"  - 因NaN丢弃样本: {dropped_samples}个 ({dropped_samples/initial_len*100:.1f}%)")
    
    return X, y, df


# ==================== 树模型配置（精简版）====================
def get_tree_models():
    """定义精选的树模型"""

    models = {
        # ==================== CatBoost (Boosting家族代表) ====================
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

        # ==================== Random Forest (Bagging家族代表) ====================
        'RF_Optimized': RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
    }

    return models


def get_model_descriptions():
    """获取每个模型的参数说明"""
    descriptions = {
        'CatBoost_Optimized': 'CatBoost-优化参数 (Boosting家族代表, 抗过拟合强)',
        'RF_Optimized': '随机森林-优化参数 (Bagging家族代表, 稳定性好)',
    }
    return descriptions


# ==================== LSTM模型定义 ====================
if HAS_TORCH:
    class LSTMModel(nn.Module):
        """LSTM时序预测模型"""
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


def prepare_lstm_data(X, y, sequence_length=24):
    """为LSTM准备序列数据"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, learning_rate=0.001):
    """训练LSTM模型"""
    if not HAS_TORCH:
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 准备数据
    sequence_length = 24
    X_train_seq, y_train_seq, scaler_X, scaler_y = prepare_lstm_data(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq, _, _ = prepare_lstm_data(X_val, y_val, sequence_length)
    
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
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
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, scaler_X, scaler_y, sequence_length


def predict_lstm(model, X, scaler_X, scaler_y, sequence_length):
    """使用LSTM模型预测"""
    if not HAS_TORCH:
        return None
    
    device = next(model.parameters()).device
    X_scaled = scaler_X.transform(X)
    
    # 构建序列
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i+sequence_length])
    X_seq = np.array(X_seq)
    
    X_tensor = torch.FloatTensor(X_seq).to(device)
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()
    
    # 反标准化
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    return predictions


# ==================== Prophet模型封装 ====================
class ProphetWrapper:
    """Prophet模型包装器，用于兼容sklearn接口"""
    
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
    
    def fit(self, X, y):
        """训练Prophet模型"""
        if not HAS_PROPHET:
            raise ImportError("Prophet未安装")
        
        # Prophet需要datetime索引
        if hasattr(X, 'index'):
            dates = X.index
        else:
            raise ValueError("Prophet需要带datetime索引的数据")
        
        # 标准化y
        self.scaler_y = MinMaxScaler()
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # 准备Prophet数据格式
        df_prophet = pd.DataFrame({
            'ds': dates,
            'y': y_scaled
        })
        
        # 创建并训练模型
        self.model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # 添加额外的回归变量（选择重要特征）
        feature_cols = ['AQI', 'hour', 'day_of_week', 'month']
        available_cols = [col for col in feature_cols if col in X.columns]
        
        for col in available_cols:
            df_prophet[col] = X[col].values
            self.model.add_regressor(col)
        
        self.model.fit(df_prophet)
        self.feature_cols = available_cols
        
        return self
    
    def predict(self, X):
        """使用Prophet模型预测"""
        if not HAS_PROPHET or self.model is None:
            raise ValueError("模型未训练或Prophet未安装")
        
        # 准备预测数据
        dates = X.index
        df_future = pd.DataFrame({'ds': dates})
        
        # 添加回归变量
        for col in self.feature_cols:
            if col in X.columns:
                df_future[col] = X[col].values
        
        # 预测
        forecast = self.model.predict(df_future)
        predictions_scaled = forecast['yhat'].values
        
        # 反标准化
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions


# ==================== 模型评估函数 ====================
def evaluate_model_cv(model, X, y, model_name, model_type='tree'):
    """通用模型评估函数（支持树模型、LSTM、Prophet）"""
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
    
    best_fold_model = None
    best_fold_r2 = -np.inf

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        try:
            if model_type == 'lstm':
                # LSTM训练
                if not HAS_TORCH:
                    print(f"  ✗ Fold {fold+1}: PyTorch未安装")
                    continue
                lstm_result = train_lstm_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                                               epochs=30, batch_size=64, learning_rate=0.001)
                if lstm_result is None:
                    continue
                fold_model, scaler_X, scaler_y, seq_len = lstm_result
                
                # 预测
                y_val_pred = predict_lstm(fold_model, X_val_fold, scaler_X, scaler_y, seq_len)
                
                # 对齐长度
                min_len = min(len(y_val_pred), len(y_val_fold))
                y_val_pred = y_val_pred[:min_len]
                y_val_fold_array = y_val_fold.iloc[:min_len].values
                
            elif model_type == 'prophet':
                # Prophet训练
                if not HAS_PROPHET:
                    print(f"  ✗ Fold {fold+1}: Prophet未安装")
                    continue
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # 预测
                y_val_pred = fold_model.predict(X_val_fold)
                
                # 对齐长度
                min_len = min(len(y_val_pred), len(y_val_fold))
                y_val_pred = y_val_pred[:min_len]
                y_val_fold_array = y_val_fold.iloc[:min_len].values
                
            else:
                # 树模型训练
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # 预测
                y_val_pred = fold_model.predict(X_val_fold)
                y_val_fold_array = y_val_fold.values
            
            # 计算指标
            mse = mean_squared_error(y_val_fold_array, y_val_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_fold_array, y_val_pred)
            r2 = r2_score(y_val_fold_array, y_val_pred)

            cv_scores['mse'].append(mse)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            
            # 保存最佳fold的模型
            if r2 > best_fold_r2:
                best_fold_r2 = r2
                best_fold_model = fold_model

            print(f"  Fold {fold+1}/5: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        except Exception as e:
            print(f"  ✗ Fold {fold+1} 失败: {str(e)}")
            continue

    elapsed_time = time.time() - start_time

    if len(cv_scores['r2']) == 0:
        print(f"  ✗ 所有fold均失败")
        return None

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
    
    # 输出特征重要性（仅树模型）
    if model_type == 'tree':
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


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("=" * 70)
    print(" 模型对比实验 - AQI预测（树模型 + 时序模型）")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  - 测试站点: {SITE_CODE}")
    print(f"  - 预测未来: {FORECAST_HOURS}小时")
    print(f"  - 滞后特征: 前{LAG_HOURS}小时")
    print(f"  - 交叉验证: 5折时间序列")
    print(f"  - PyTorch可用: {HAS_TORCH}")
    print(f"  - Prophet可用: {HAS_PROPHET}")

    # 加载数据
    X, y, df_original = load_and_prepare_data(SITE_CODE)

    if X is None:
        print("\n❌ 数据加载失败，程序退出")
        exit(1)

    # ==================== 获取模型 ====================
    tree_models = get_tree_models()
    descriptions = get_model_descriptions()

    all_models = {}
    all_descriptions = {}
    
    # 添加树模型
    for name, model in tree_models.items():
        all_models[name] = (model, 'tree')
        all_descriptions[name] = descriptions.get(name, '')
    
    # 添加LSTM模型
    if HAS_TORCH:
        all_models['LSTM'] = (None, 'lstm')
        all_descriptions['LSTM'] = '长短期记忆网络 (深度学习时序模型)'
    else:
        print("\n⚠️ 跳过LSTM模型（PyTorch未安装）")
    
    # 添加Prophet模型
    if HAS_PROPHET:
        all_models['Prophet'] = (ProphetWrapper(), 'prophet')
        all_descriptions['Prophet'] = 'Facebook Prophet (统计时序模型)'
    else:
        print("\n⚠️ 跳过Prophet模型（Prophet未安装）")

    print(f"\n📋 将要评估 {len(all_models)} 个模型:")
    for i, (name, desc) in enumerate(all_descriptions.items(), 1):
        print(f"  {i:2d}. {name:<20s} - {desc}")

    # ==================== 评估所有模型 ====================
    all_results = []

    for model_name, (model, model_type) in all_models.items():
        result = evaluate_model_cv(model, X, y, model_name, model_type=model_type)
        
        if result is not None:
            result['description'] = all_descriptions.get(model_name, '')
            all_results.append(result)

            # 保存模型（树模型才保存）
            if model_type == 'tree' and model is not None:
                model_path = os.path.join(output_dir, f'{model_name.lower()}_model.pkl')
                joblib.dump(model, model_path)
                print(f"  ✓ 模型已保存: {model_path}\n")
        else:
            print(f"  ✗ {model_name} 评估失败，跳过\n")

    # ==================== 汇总结果 ====================
    if len(all_results) == 0:
        print("\n❌ 没有模型成功评估")
        exit(1)

    print("\n\n" + "=" * 70)
    print("📊 模型对比结果汇总")
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
    summary_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 对比结果已保存: {summary_path}")

    # 找出最佳模型
    best_model = results_df.iloc[0]
    print(f"\n🏆 最佳模型: {best_model['model_name']}")
    print(f"   说明: {best_model['description']}")
    print(f"   R² = {best_model['r2_mean']:.4f} ± {best_model['r2_std']:.4f}")
    print(f"   RMSE = {best_model['rmse']:.2f}")
    print(f"   MAE = {best_model['mae']:.2f}")

    # 按模型类型分类分析
    print("\n" + "=" * 70)
    print("📈 按模型类型分类对比")
    print("=" * 70)

    model_types = {
        '树模型 (Tree Ensemble)': ['CatBoost_Optimized', 'RF_Optimized'],
        '深度学习 (Deep Learning)': ['LSTM'],
        '统计模型 (Statistical)': ['Prophet']
    }

    for type_name, model_names in model_types.items():
        type_results = results_df[results_df['model_name'].isin(model_names)]
        if len(type_results) > 0:
            best_in_type = type_results.iloc[0]
            avg_r2 = type_results['r2_mean'].mean()
            print(f"\n🌟 {type_name}:")
            print(f"   最佳: {best_in_type['model_name']} (R²={best_in_type['r2_mean']:.4f})")
            print(f"   平均: R²={avg_r2:.4f}")
            print(f"   模型数: {len(type_results)}")

    # Top 推荐
    print("\n" + "=" * 70)
    print("🎯 最终推荐")
    print("=" * 70)
    top3 = results_df.head(3)
    for idx, row in top3.iterrows():
        print(f"\n  #{idx+1}: {row['model_name']}")
        print(f"      {row['description']}")
        print(f"      R² = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
        print(f"      RMSE = {row['rmse']:.2f}, MAE = {row['mae']:.2f}")

    print("\n✅ 模型对比完成！")
