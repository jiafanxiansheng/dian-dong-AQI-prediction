import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import warnings
import time

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

FORECAST_HOURS = 12
SEQUENCE_LENGTH = 24

output_dir = r'C:\Users\28927\dazuoye\pythonProject3\模型'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 模型保存目录: {output_dir}\n")

engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


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
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


def prepare_lstm_data(df, sequence_length=SEQUENCE_LENGTH):
    """为LSTM准备序列数据"""
    df_feat = df.copy()

    base_features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO', 'AQI']
    available_features = [f for f in base_features if f in df_feat.columns]

    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat.index.hour / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat.index.hour / 24)
    available_features += ['hour_sin', 'hour_cos']

    target_col = f'AQI_future_{FORECAST_HOURS}h'
    df_feat[target_col] = df_feat['AQI'].shift(-FORECAST_HOURS)

    df_feat = df_feat.dropna()

    feature_data = df_feat[available_features].values
    target_data = df_feat[target_col].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    feature_scaled = scaler_X.fit_transform(feature_data)
    target_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()

    X_sequences, y_sequences = [], []
    for i in range(len(feature_scaled) - sequence_length):
        X_sequences.append(feature_scaled[i:i + sequence_length])
        y_sequences.append(target_scaled[i + sequence_length])

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    return X_seq, y_seq, scaler_X, scaler_y, available_features


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, learning_rate=0.001):
    """训练单个LSTM模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"      Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_lstm(model, X_seq, scaler_y):
    """使用LSTM模型预测"""
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(X_seq).to(device)

    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()

    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    return predictions


def train_site_model(site_name):
    """为单个站点训练LSTM模型（Hold-out验证版）"""
    print("\n" + "=" * 70)
    print(f"开始训练站点 {site_name} 的LSTM模型")
    print("=" * 70)

    start_time = time.time()

    table_name = f'air_quality_site_{site_name.lower()}'

    query = f"SELECT * FROM `{table_name}` ORDER BY `datetime`"
    df = pd.read_sql(query, engine)
    print(f"✓ 成功读取 {len(df)} 条数据")

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

    X_seq, y_seq, scaler_X, scaler_y, features = prepare_lstm_data(df, SEQUENCE_LENGTH)
    print(f"\n✓ 序列数据准备完成:")
    print(f"  - 序列长度: {SEQUENCE_LENGTH}")
    print(f"  - 特征维度: {len(features)}")
    print(f"  - 样本数量: {len(X_seq)}")

    # ✅ Hold-out验证：80%训练，20%测试
    train_size = int(len(X_seq) * 0.8)
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_test = X_seq[train_size:]
    y_test = y_seq[train_size:]

    print(f"\n📊 Hold-out验证分割:")
    print(f"  - 训练集: {len(X_train)} 样本 ({train_size / len(X_seq) * 100:.0f}%)")
    print(f"  - 测试集: {len(X_test)} 样本 ({(len(X_seq) - train_size) / len(X_seq) * 100:.0f}%)")

    print(f"\n  训练LSTM模型...")
    model = train_lstm_model(
        X_train, y_train, X_test, y_test,
        epochs=50, batch_size=64, learning_rate=0.001
    )

    # 评估
    test_pred = predict_lstm(model, X_test, scaler_y)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_actual, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    mae = mean_absolute_error(y_test_actual, test_pred)

    mask = y_test_actual > 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_test_actual[mask], test_pred[mask])
    else:
        mape = np.nan

    if np.isnan(mape):
        print(f"\n✓ 测试集性能: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE=N/A")
    else:
        print(f"\n✓ 测试集性能: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2%}")

    # ✅ 直接保存评估过的模型（不再重新训练）
    model_path = os.path.join(output_dir, f'aqi_lstm_model_{site_name}_future{FORECAST_HOURS}h.pkl')
    joblib.dump({
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': features,
        'sequence_length': SEQUENCE_LENGTH
    }, model_path)

    elapsed_time = time.time() - start_time

    print(f"✓ 模型已保存: {model_path}")
    print(f"✓ 训练耗时: {elapsed_time:.2f}秒")

    return {
        'site': site_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape if not np.isnan(mape) else None,
        'samples': len(X_seq),
        'time': elapsed_time,
        'model_path': model_path
    }


if __name__ == '__main__':
    print("=" * 70)
    print("开始批量训练所有站点的LSTM AQI预测模型（未来12小时）")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  - 预测未来: {FORECAST_HOURS}小时")
    print(f"  - 序列长度: {SEQUENCE_LENGTH}小时")
    print(f"  - 站点数量: {len(target_sites)}个")
    print(f"  - GPU可用: {torch.cuda.is_available()}")
    print(f"  - 验证方式: Hold-out (80/20)")
    print(f"  - 训练策略: 单次训练（节省时间）")

    all_results = []

    for i, site_name in enumerate(target_sites, 1):
        print(f"\n[{i}/{len(target_sites)}] ", end="")
        result = train_site_model(site_name)
        if result is not None:
            all_results.append(result)

    print("\n\n" + "=" * 70)
    print("所有站点LSTM模型训练完成！汇总结果:")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n模型性能对比:")
        print("-" * 80)
        print(
            f"{'站点':<10} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10} | {'MAPE':<10} | {'样本数':<10} | {'耗时(s)':<8}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            mape_str = f"{row['mape']:.2%}" if row['mape'] is not None else "N/A"
            print(
                f"{row['site']:<10} | {row['r2']:<10.4f} | {row['rmse']:<10.2f} | {row['mae']:<10.2f} | {mape_str:<10} | {row['samples']:<10} | {row['time']:<8.2f}")

        print("-" * 80)
        mape_mean = results_df['mape'].dropna().mean()
        mape_str = f"{mape_mean:.2%}" if not np.isnan(mape_mean) else "N/A"
        print(
            f"{'平均':<10} | {results_df['r2'].mean():<10.4f} | {results_df['rmse'].mean():<10.2f} | {results_df['mae'].mean():<10.2f} | {mape_str:<10}")

        summary_path = os.path.join(output_dir, 'lstm_model_training_summary_12h_holdout.csv')
        results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 汇总结果已保存: {summary_path}")

        best_model_idx = results_df['r2'].idxmax()
        best_model = results_df.loc[best_model_idx]
        print(f"\n🏆 最佳LSTM模型: 站点{best_model['site']}, R²={best_model['r2']:.4f}")
    else:
        print("\n✗ 没有成功训练的模型")

    print("\n✅ 所有任务完成！")
