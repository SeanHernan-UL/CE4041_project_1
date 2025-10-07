
# following the guide at https://www.tensorflow.org/tutorials/quickstart/beginner
# and https://learnopencv.com/implementing-cnn-tensorflow-keras/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import * # import everything cos lazy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

print("beep boop")
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

# getting mnist data
(training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()

## preprocessing
# converting images to float 32 and adding fourth axis
training_images= (training_inputs.astype('float32') / 255.0)[:,:,:,np.newaxis]
test_images =  (test_inputs.astype('float32') / 255.0)[:,:,:,np.newaxis]
# converting to 'one-hot' encoding
categorical_training_labels  = to_categorical(training_labels)
categorical_test_labels = to_categorical(test_labels)

print(f'training_images.shape = {training_images.shape}')
print(f'test_images.shape = {test_images.shape}')

# plt.figure(figsize=(28,28))
#
# num_rows = 10
# num_cols = 10

## plot first 100 images
# for i in range(num_rows*num_cols):
#     ax = plt.subplot(num_rows, num_cols, i + 1)
#     plt.imshow(training_inputs[i,:,:], cmap='gray', vmin=0, vmax=1)
#     plt.axis("off")
#
# plt.show()

## this is horrible...
# print('After conversion to categorical one-hot encoded labels: ', categorical_training_outputs[0])
# print('After conversion to categorical one-hot encoded labels: ', categorical_training_outputs[1])
# print('After conversion to categorical one-hot encoded labels: ', categorical_training_outputs[2])

# adding training parameters
# values used need to be justified...
EPOCHS = 30
SPLIT = 0.2
SHUFFLE = True
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = 'RMSprop'

# potentially need to initialise and set a random seed
# it might do this anyways??

input_shape = training_images.shape[1:]

# create the model
model = tf.keras.models.Sequential()
model.add(Input(shape=input_shape))
model.add(layers.Conv2D(32, 5, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

# just use one conv layer for now
model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# adding final layers to do classification
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=OPTIMIZER,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                     verbose=2, mode='auto',
                     restore_best_weights=True)

history = model.fit(training_images, categorical_training_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE, verbose=2,callbacks=[stop], validation_split=SPLIT)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  categorical_test_labels, verbose=2)

print(test_acc)

###################################################################

# We can also test the performance of the network on the completely separate
# testing set.  We get about 98% accuracy.  Note that this is testing on
# unseen inputs, so is a true measure of performance.
#


print("Performance of network on testing set:")
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))


# It is also interesting to see the accuracy reported on the training and
# validation data.  The highest accuracy is always reported by the
# training set, validation is worse, and typically better than the accuracy
# reported by testing on the unseen testing set.  Here validation and testing
# accuracies are about the same.
#

print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))


# Suggestions: Try increasing the number of convolutional layers and the
# number of slices in the deeper convolutional layers (with maxpooling
# between the layers for dimensionality reduction).  Consider adding
# (judiciously) one or more Dropout layers.
#
plt.show()