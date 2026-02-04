import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

scaler_x = joblib.load("scalers/scaler_x.pkl")
scaler_y = joblib.load("scalers/scaler_y.pkl")

df_test = pd.read_csv("train/DailyDelhiClimateTest.csv")

df_test_fix = df_test[[
    "meantemp",
    "humidity",
    "wind_speed",
    "meanpressure"
]].copy()

df_test_fix["temp_target"] = df_test_fix["meantemp"].shift(-1)
df_test_fix.loc[df_test_fix.index[-1], "temp_target"] = (
    df_test_fix.loc[df_test_fix.index[-1], "meantemp"]
)

x_test = df_test_fix.drop(columns=["temp_target"])
y_test = df_test_fix["temp_target"]


x_test_scaled = scaler_x.transform(x_test)

MODEL_PATH = "models/model_tf"

tfsm_layer = keras.layers.TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

inputs = keras.Input(shape=(4,), name="input")
outputs = tfsm_layer(inputs)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

loss, mae= model.evaluate(x_test_scaled, scaler_y.transform(y_test.values.reshape(-1, 1)))

y_pred_scaled = model.predict(x_test_scaled)
if isinstance(y_pred_scaled, dict):
    y_pred_scaled = next(iter(y_pred_scaled.values()))
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = y_test.values.reshape(-1, 1)

r2 = r2_score(y_true, y_pred)


with mlflow.start_run():
   
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    
    mlflow.keras.log_model(
        model,
        name="model"
    )

    mlflow.log_artifact("scalers/scaler_x.pkl")
    mlflow.log_artifact("scalers/scaler_y.pkl")

    print("Model & metrics berhasil di-log ke MLflow")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")
