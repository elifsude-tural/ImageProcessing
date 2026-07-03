from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(figsize=(14,14))
x, y = 10, 4
for i in range(40):
  plt.subplot(x,y, i+1)
  plt.imshow(x_train[i])
plt.show()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

batch_size = 128
num_classes = 10
epochs = 5

img_rows, img_cols = 28,28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation = 'relu',
                 input_shape= input_shape))
model.add(Conv2D(64, kernel_size=(3,3),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation = "softmax"))

model.summary()

model.compile(loss= keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

(_, original_y_train), (_, original_y_test) = mnist.load_data()

# Convert class vectors to binary class matrices using the correct original labels
y_train_corrected = keras.utils.to_categorical(original_y_train, num_classes)
y_test_corrected = keras.utils.to_categorical(original_y_test, num_classes)

model.fit(x_train, y_train_corrected,
          batch_size=batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data=(x_test, y_test_corrected))

print(y_train.shape)
print(y_test.shape)
