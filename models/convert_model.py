import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("model.keras")

# model.export("model_tf")

print("Model converted to TensorFlow SavedModel format")