import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
def calculate_norm_eq(X, y):
    tmp = X.T.dot(X)
    tmp = np.linalg.pinv(tmp)
    tmp = tmp.dot(X.T)
    theta_hat = tmp.dot(y)
    print(f"theta_hat: {theta_hat} and {theta_hat.shape}")
    return theta_hat

def calculate_hypothesis(X, theta):
    y = theta.T.dot(X.T)
    return y

def create_window(X, window_size):
    windows = []
    for i in range(len(X)-window_size):
        window = X.iloc[i : i + window_size].values.flatten()
        window = np.nan_to_num(window, nan=0.5)
        windows.append(window)
    return windows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument("--observation_window", type=int, required=True)
    parser.add_argument("--predicted_currency", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    target_column = f"Low: coin_{args.predicted_currency}"
    if target_column not in df.columns:
        raise ValueError(f"Currency column '{target_column}' not found in dataset")
    y = df[target_column]
    print(y.isna().sum())
    y = y.interpolate().fillna(method='bfill').fillna(method='ffill')
    X = df.drop(columns=[target_column])
    X_windowed = create_window(X, args.observation_window)
    X_windowed = np.array(X_windowed)
    #scaler = StandardScaler()
    #X_windowed = scaler.fit_transform(X_windowed)
    y_trimmed = y[args.observation_window:].values
    #print("Any NaNs in y_trimmed?", np.isnan(y_trimmed).any())
    #print("Variance of y_trimmed:", np.var(y_trimmed))

    theta = calculate_norm_eq(X_windowed, y_trimmed)
    predictions = calculate_hypothesis(X_windowed, theta)
    print(f"Predictions: {predictions}")

    plt.plot(y_trimmed, label="Actual Prices", color='blue')
    plt.plot(predictions, label="Predicted Prices", color='red')
    plt.xlabel('Observation')
    plt.ylabel('Price')
    plt.legend()
    plt.title(f"Actual vs Predicted: {args.predicted_currency.capitalize()}")
    plt.savefig('prediction.png', bbox_inches='tight')
    mse = np.mean((predictions - y_trimmed) ** 2)
    print(f"Mean Squared Error: {mse}")


