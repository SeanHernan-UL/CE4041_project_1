# kanvar murray
# 22374698
# 03/11/2025

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import logging
import sys

# setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info("Starting MNIST CNN repeated training script")

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

# list to store accuracies of each run
all_accuracies = []

# initialize plot
plt.figure(figsize=(10,6))

# loop over 10 runs
for run in range(1, 11):
    print(f"\nRun {run} -----------------------------")
    logger.info(f"Run {run} started")

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
        verbose=0
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
        verbose=0  # hide batch-by-batch output
    )

    # plot accuracy for this run
    plt.plot(history.history['val_accuracy'], label=f'Run {run}')

    # evaluate on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Training ran for {len(history.history['accuracy'])} epochs")
    all_accuracies.append(test_acc*100)

# finalize plot
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Over 10 Runs')
plt.legend(loc='lower right', fontsize=8)
plt.show()

# print all 10 accuracies and average
print("\nAll 10 test accuracies:")
for i, acc in enumerate(all_accuracies, 1):
    print(f"Run {i}: {acc:.2f}%")
print(f"Average Test Accuracy over 10 runs: {np.mean(all_accuracies):.2f}%")
logger.info(f"All 10 test accuracies: {all_accuracies}")
logger.info(f"Average Test Accuracy: {np.mean(all_accuracies):.2f}%")
