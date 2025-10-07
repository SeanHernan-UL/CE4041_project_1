
# following the guide at https://www.tensorflow.org/tutorials/quickstart/beginner
# and https://learnopencv.com/implementing-cnn-tensorflow-keras/

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import *
from tensorflow.keras.utils import to_categorical

print("beep boop")
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f'x_train.shape = {x_train.shape}')
print(f'x_test.shape = {x_test.shape}')

# plt.figure(figsize=(28,28))
#
# num_rows = 10
# num_cols = 10

# # plot first 100 images
# for i in range(num_rows*num_cols):
#     ax = plt.subplot(num_rows, num_cols, i + 1)
#     plt.imshow(x_train[i,:,:], cmap='gray', vmin=0, vmax=1)
#     plt.axis("off")
#
# plt.show()

# # Change the labels from integer to categorical data.
# print('Original (integer) label for the first training sample: ', y_train[0])
# print('Original (integer) label for the first training sample: ', y_train[1])
# print('Original (integer) label for the first training sample: ', y_train[2])
#
# # Convert labels to one-hot encoding.
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
# # this is horrible...
# print('After conversion to categorical one-hot encoded labels: ', y_train[0])
# print('After conversion to categorical one-hot encoded labels: ', y_train[1])
# print('After conversion to categorical one-hot encoded labels: ', y_train[2])

# adding training parameters
epochs = 5
batch_size = 128
learning_rate = 0.001

# create a model??
model = tf.keras.models.Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()

# adding final layers to do classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(test_acc)

###################################################################
