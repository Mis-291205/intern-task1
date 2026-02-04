import numpy as np
import pandas as pd
import joblib
import keras

class TemperaturePredictor:
    def __init__(
        self,
        model_path="models/model_tf",
        scaler_x_path="scalers/scaler_x.pkl",
        scaler_y_path="scalers/scaler_y.pkl"
    ):
        print("Loading model and scalers...")

        self.model = keras.layers.TFSMLayer(
            model_path,
            call_endpoint="serving_default"
        )

        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

        self.feature_names = [
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure"
        ]

        print("Model and scalers loaded successfully!")

    def preprocess(self, input_data: dict):
        df = pd.DataFrame([input_data], columns=self.feature_names)
        scaled = self.scaler_x.transform(df)
        return scaled.astype(np.float32)

    def predict(self, input_data: dict):

        scaled_input = self.preprocess(input_data)

        pred_output = self.model(scaled_input)

        if isinstance(pred_output, dict):
            pred_output = list(pred_output.values())[0]

        pred_scaled = pred_output.numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled)

        return float(pred[0][0])

if __name__ == "__main__":

    predictor = TemperaturePredictor()

    print("\nInput data to predict next day temperature")

    input_dict = {}

    for feature in predictor.feature_names:
        val = float(input(f"{feature}: "))
        input_dict[feature] = val

    result = predictor.predict(input_dict)

    print(f"\nPredicted next day temperature: {result:.2f}")
