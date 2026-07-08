import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = tf.keras.models.load_model("saved_models/digit_model.keras")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "data/val",
    image_size=(28,28),
    color_mode="grayscale",
    batch_size=32, 
    label_mode="int",
    shuffle=False
)

val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

predictions = model.predict(val_dataset)
y_pred = np.argmax(predictions, axis=1)


y_true = []

for images, labels in val_dataset:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
