import tensorflow as tf
import numpy as np
from tensorflow import keras
from data_loader import load_datasets

model = tf.keras.models.load_model("saved_models/digit_model.keras")

_, val_ds = load_datasets()


loss, accuracy = model.evaluate(val_ds)
print(f"validation loss: {loss:.4f}")
print(f"validation accuracy: {accuracy:.4f}")
