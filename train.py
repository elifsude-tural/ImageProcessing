from tensorflow import keras
from model import build_model
from data_loader import load_datasets


batch_size = 128
epochs = 5

train_ds, val_ds = load_datasets(batch_size=batch_size)

model = build_model()
model.summary()

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=epochs,
    verbose=1,
    validation_data=val_ds
)
