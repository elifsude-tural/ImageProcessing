import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array




IMAGE_PATH = r"data\val\0\00003.png"
saved_model = tf.keras.models.load_model("saved_models/digit_model.keras")

img = load_img(
    IMAGE_PATH,
    color_mode="grayscale",
    target_size=(28, 28)
)


img_array = img_to_array(img)
img_array = img_array/255.0
img_array = img_array.reshape(1, 28, 28, 1)


prediction = saved_model.predict(img_array)

predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])

print("tahmin edilen rakam:", predicted_class)
print("guven orani", confidence)
