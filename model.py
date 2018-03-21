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

def load_log(path, images, steeringAngles):
    dirPath = os.path.dirname(path)
    print("Loading images from {}".format(dirPath))
    with open(path) as drivingLog:
        reader = csv.reader(drivingLog)
        for line in reader:
            angle = float(line[3])

            centerImagePath = convert_image_path(dirPath, line[0])
            centerImage = cv2.imread(centerImagePath)
            images.append(centerImage)
            steeringAngles.append(angle)

            leftImagePath = convert_image_path(dirPath, line[1])
            leftImage = cv2.imread(leftImagePath)
            images.append(leftImage)
            steeringAngles.append(angle + 0.2)

            rightImagePath = convert_image_path(dirPath, line[2])
            rightImage = cv2.imread(rightImagePath)
            images.append(rightImage)
            steeringAngles.append(angle - 0.2)

def load_training_data(dirs):
    images = list()
    steeringAngles = list()
    for directory in dirs:
        load_log(os.path.join(directory, "driving_log.csv"),
            images, steeringAngles)
    return numpy.array(images), numpy.array(steeringAngles)

def rgb_to_gray(x):
    return 0.3 * x[:,:,:,0:1] + 0.59 * x[:,:,:,1:2] + 0.11 * x[:,:,:,-1:]

def normalize(x):
    return x / 255.0 - 0.5

trainX, trainY = load_training_data([
    "data/3-1l-ccw",
    "data/4-1l-cw",
])

model = Sequential()

model.add(Cropping2D(
    cropping=((50, 25), (0, 0)),
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

adam = Adam(lr=0.001, decay=0.1)
model.compile(loss="mse", optimizer=adam)

modelCheckpoint = ModelCheckpoint(
    filepath="models/model-{val_loss:.4f}-{loss:.4f}-{epoch:02d}.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True)
earlyStopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10)
model.fit(trainX, trainY,
    callbacks=[
        modelCheckpoint,
        earlyStopping,
    ],
    validation_split=0.2,
    shuffle=True,
    epochs=100)
