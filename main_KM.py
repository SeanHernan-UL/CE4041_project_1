# kanvar murray
# 22374698
# 03/11/2025

import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
from datetime import datetime
import os
import logging
import sys

# create folder for test run
test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"./mnist_training_{test_time}/"
os.makedirs(path, exist_ok=True)

# create folder for csv
csv_path = f"./mnist_training_{test_time}/csv/"
os.makedirs(csv_path, exist_ok=True)

# create folder for plots
image_path = f"./mnist_training_{test_time}/plots/"
os.makedirs(image_path, exist_ok=True)

# setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f'{path}/mnist_training_{test_time}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info("Starting MNIST CNN repeated training script")
logging.info(f"Saving data to {path}")

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

# initialize plots: train, validation
plt.figure(1,figsize=(10,6))
plt.figure(2,figsize=(10,6))

# will be list of lists -> Dataframe -> csv file
csv_labels = ["training", "validation", "test"]
csv_data = [[],[],[]]

logger.info(f"Logger only grabbing max values per run (see .csv for full data)")
# loop over 10 runs
test_range = 3
for run in range(1, test_range+1):
    logger.info(f"Run {run} -----------------------------")
    logger.info(f"Run, Train, Validation, Test")

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

    # TODO pipe tensorflow console output also to log file

    # evaluate on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)

    # grab test run data
    training = np.array(history.history['accuracy'])*100
    validation = np.array(history.history['val_accuracy'])*100
    test = test_acc*100

    # plot the data
    plt.figure(1)
    plt.plot(training, label=f'Run {run}')
    plt.figure(2)
    plt.plot(validation, label=f'Run {run}')

    # save + log to a file
    csv_data[0].append(training)
    csv_data[1].append(validation)
    csv_data[2].append(test)
    logger.info("{}, {:6.3f}, {:6.3f}, {:6.3f}".format(run,training[-1], validation[-1], test))

    print(f"Training ran for {len(history.history['accuracy'])} epochs")
    all_accuracies.append(test_acc*100)

logging.info("Saving csv files")
# create dataframe from csv_data
for i in range(0,3):
    df = pd.DataFrame(csv_data[i])
    df.to_csv(f"{csv_path}/mnist_{csv_labels[i]}_{test_time}.csv", index=False)

# finalize plot
logging.info("Saving plots as png files")
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy (%)')
plt.title(f'Training Accuracy Over {test_range} Runs')
plt.legend(loc='lower right', fontsize=8)
plt.savefig(f"{image_path}/mnist_train_{test_time}.png")

plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title(f'Validation Accuracy Over {test_range} Runs')
plt.legend(loc='lower right', fontsize=8)
plt.savefig(f"{image_path}/mnist_validation_{test_time}.png")

# print all 10 accuracies and average
print(f"\nAll {test_range} test accuracies:")
for i, acc in enumerate(all_accuracies, 1):
    print(f"Run {i}: {acc:.2f}%")
print(f"Average Test Accuracy over 10 runs: {np.mean(all_accuracies):.2f}%")
logger.info(f"All 10 test accuracies: {all_accuracies}")
logger.info(f"Average Test Accuracy: {np.mean(all_accuracies):.2f}%")
