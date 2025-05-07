import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# 参数设置
LOOK_BACK = 24
BATCH_SIZE = 128
EPOCHS = 30
MODEL_PATH = 'best_lstm_model.keras'
DATA_PATH = 'LSTM-Multivariate_pollution.csv'

def load_and_preprocess_data(path):
    df = pd.read_csv(path, parse_dates=['date'], dayfirst=True)
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['pollution']).reset_index(drop=True)

    # 编码风向
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    wind_encoded = encoder.fit_transform(df[['wnd_dir']])

    # 数值特征
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']])

    # 合并特征
    processed_data = np.concatenate([num_features, wind_encoded], axis=1)
    return processed_data, scaler, encoder

def create_sequences(data, target_index, look_back):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, target_index])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def inverse_pm25_scaling(values, scaler):
    dummy = np.zeros((len(values), 7))  # 数值特征数量为7
    dummy[:, 0] = values  # PM2.5是第一列
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def plot_predictions(true_vals, pred_vals):
    plt.figure(figsize=(14, 6))
    plt.plot(true_vals[:500], label='True PM2.5', linewidth=1)
    plt.plot(pred_vals[:500], label='Predicted PM2.5', linestyle='--', linewidth=1)
    plt.title('PM2.5 Forecasting (First 500 Samples)')
    plt.xlabel('Hour')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists(DATA_PATH):
        print(f"数据文件未找到，请确认路径: {DATA_PATH}")
        return

    # 加载并预处理数据
    data, scaler, encoder = load_and_preprocess_data(DATA_PATH)
    target_index = 0  # pollution

    # 构建时序样本
    X, y = create_sequences(data, target_index, LOOK_BACK)

    # 时间连续划分训练/测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建模型
    model = build_lstm_model((LOOK_BACK, X.shape[2]))

    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # 测试评估
    loss, mae = model.evaluate(X_test, y_test)
    print(f"\nTest MSE: {loss:.4f}, Test MAE: {mae:.4f}")

    # 预测 + 逆标准化
    predictions = model.predict(X_test).ravel()
    y_true_orig = inverse_pm25_scaling(y_test, scaler)
    y_pred_orig = inverse_pm25_scaling(predictions, scaler)

    # 可视化
    plot_predictions(y_true_orig, y_pred_orig)

if __name__ == '__main__':
    main()
