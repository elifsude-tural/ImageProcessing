from tensorflow import keras
from model import build_model
from data_generator import DigitDataGenerator
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
""" 
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

"""
loss_fn   = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

train_acc = keras.metrics.SparseCategoricalAccuracy()
val_acc   = keras.metrics.SparseCategoricalAccuracy()
val_loss_metric = keras.metrics.Mean()   

import tensorflow as tf

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)     
        loss  = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y, preds)
    return loss

@tf.function
def val_step(x, y):
    preds = model(x, training=False)     
    loss  = loss_fn(y, preds)
    val_acc.update_state(y, preds)
    val_loss_metric.update_state(loss)

for epoch in range(epochs):
    train_acc.reset_state()
    val_acc.reset_state()
    val_loss_metric.reset_state()

    for i in range(len(train_gen)):
        x_batch, y_batch = train_gen[i]
        train_step(x_batch, y_batch)

    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        val_step(x_batch, y_batch)

    train_gen.on_epoch_end()   

    print(f"Epoch {epoch+1}/{epochs} - "
          f"train_acc: {train_acc.result():.4f} - "
          f"val_acc: {val_acc.result():.4f} - "
          f"val_loss: {val_loss_metric.result():.4f}")

os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/digit_model.keras")
