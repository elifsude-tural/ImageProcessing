from tensorflow import keras
from model import build_model
from data_loader import load_datasets


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
    validation_data=val_gen
)
