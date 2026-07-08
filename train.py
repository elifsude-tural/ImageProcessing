from tensorflow import keras
from model import build_model
from data_loader import load_datasets
import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

tensorboard_callback = TensorBoard(log_dir="logs")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "saved_models/best_digit_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

batch_size = 128
epochs = 5

train_gen = DigitDataGenerator("data/train", batch_size=batch_size, augment=True, shuffle=True)
val_gen = DigitDataGenerator("data/val", batch_size=batch_size, augment=False, shuffle=False)

model = build_model()
model.summary()

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=val_gen,
    callbacks=[tensorboard_callback, early_stop, checkpoint]
)

os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/digit_model.keras")
