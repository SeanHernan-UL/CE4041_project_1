
# the guides at https://www.tensorflow.org/tutorials/quickstart/beginner
# and https://learnopencv.com/implementing-cnn-tensorflow-keras/ were used

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## do one cycle of train/validate/test, print results + generate confusion matrix

# print group + members names/ids
print('## Group 1')
members = [('Kanvar Murray', 22374698),('Seán Hernan', 22348948), ('Madeline Ware', 21306591)]
for member in members:
    print(f'{member[0]} : {member[1]}')
print('')

# load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalize and add channel dimension
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# one-hot encode labels
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# hyperparameters
EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

input_shape = train_images.shape[1:]

# build CNN model
model = models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=2
)

# train model
history = model.fit(
    train_images,
    train_labels_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop],
    verbose=2  # hide batch-by-batch output
)

# evaluate on test data
test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)

# grab test run data
training = np.array(history.history['accuracy'])*100
validation = np.array(history.history['val_accuracy'])*100
test = test_acc*100

# spit out data to console
print(f'training: {training}')
print(f'validation: {validation}')
print(f'test: {test}')

# plot raw validation data
plt.figure(1)
plt.plot(validation)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title(f'Validation Accuracy Plot')

# generate confusion matrix
responses = model.predict(test_images)
predicted_labels = np.argmax(responses, axis=1)

confusion_matrix = np.zeros([10,10],dtype=np.uint16)

for i in range(0,len(predicted_labels)):
    confusion_matrix[predicted_labels[i]][test_labels[i]] += 1

print(confusion_matrix)

plt.figure(2)
sns.heatmap(confusion_matrix, annot=True, fmt='4d', square=True)
plt.xlabel('Actual Value')
plt.ylabel('Predicted value')
plt.title(f'Confusion Matrix')

plt.show()
