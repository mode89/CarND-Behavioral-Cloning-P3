import csv
import cv2
import numpy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.optimizers import Adam
import os

def convert_image_path(dirPath, imagePath):
    return os.path.join(dirPath, "IMG", os.path.basename(imagePath))

class Sample:

    def __init__(self, dirPath, line):
        self.centerImagePath = convert_image_path(dirPath, line[0])
        self.leftImagePath = convert_image_path(dirPath, line[1])
        self.rightImagePath = convert_image_path(dirPath, line[2])
        self.steeringAngle = line[3]

def read_log(path):
    print("Reading log {} ...".format(path))
    dirPath = os.path.dirname(path)
    samples = list()
    with open(path) as drivingLog:
        reader = csv.reader(drivingLog)
        for line in reader:
            sample = Sample(dirPath, line)
            samples.append(sample)
    return samples

def read_logs(dirs):
    samples = list()
    for directory in dirs:
        samples += read_log(os.path.join(directory, "driving_log.csv"))
    return samples

def rgb_to_gray(x):
    return 0.3 * x[:,:,:,0:1] + 0.59 * x[:,:,:,1:2] + 0.11 * x[:,:,:,-1:]

def normalize(x):
    return x / 255.0 - 0.5

def create_model():
    model = Sequential()

    model.add(Cropping2D(
        cropping=((70, 25), (0, 0)),
        input_shape=(160, 320, 3)))
    model.add(Lambda(rgb_to_gray))
    model.add(Lambda(normalize))

    model.add(Conv2D(24, (5, 5),
        padding="valid", activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5),
        padding="valid", activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5),
        padding="valid", activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3),
        padding="valid", activation="relu", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3),
        padding="valid", activation="relu", strides=(1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    return model

def compile_model(model):
    adam = Adam(lr=0.001, decay=0.01)
    model.compile(loss="mse", optimizer=adam)

def train_model(model):
    if not os.path.exists("models"):
        os.mkdir("models")

    modelCheckpoint = ModelCheckpoint(
        filepath="models/model-{val_loss:.4f}-{loss:.4f}-{epoch:02d}.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True)

    earlyStopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=10)

    model.fit(trainX, trainY,
        callbacks=[
            modelCheckpoint,
            earlyStopping,
        ],
        batch_size=128,
        validation_split=0.2,
        shuffle=True,
        epochs=100)

if __name__ == "__main__":
    samples = read_logs([
        "data/10-1l-ccw",
        "data/11-1l-cw",
        "data/12-10-2c-ccw",
        "data/13-10-3c-ccw",
        "data/14-10-2c-ccw",
        "data/15-10-3c-ccw",
    ])
