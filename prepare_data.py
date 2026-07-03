import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 1. Veriyi indir
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# 2. Klasörleri oluştur
for split in ["train", "val"]:
    for digit in range(10):
        os.makedirs(f"data/{split}/{digit}", exist_ok=True)

# 3. Görüntüleri kaydet
def save_images(x, y, split):
    n = len(x)
    for i in range(n):
        img = Image.fromarray(x[i])
        label = y[i]
        img.save(f"data/{split}/{label}/{i:05d}.png")
    print(f"{split} icin {n} gorsel kaydedildi.")

save_images(x_train, y_train, "train")
save_images(x_val, y_val, "val")