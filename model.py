import csv
import cv2
import numpy
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

def load_log(path, images, steeringAngles):
    with open(path) as drivingLog:
        reader = csv.reader(drivingLog)
        for line in reader:
            angle = float(line[3])

            centerImage = cv2.imread(line[0])
            images.append(centerImage)
            steeringAngles.append(angle)

            leftImage = cv2.imread(line[1])
            images.append(leftImage)
            steeringAngles.append(angle + 0.2)

            rightImage = cv2.imread(line[2])
            images.append(rightImage)
            steeringAngles.append(angle - 0.2)

def load_training_data():
    images = list()
    steeringAngles = list()
    load_log("data/3-1l-ccw/driving_log.csv", images, steeringAngles)
    load_log("data/4-1l-cw/driving_log.csv", images, steeringAngles)
    return numpy.array(images), numpy.array(steeringAngles)

trainX, trainY = load_training_data()

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 30), (0, 0))))

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
model.add(Dense(1164, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))

model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(trainX, trainY, validation_split=0.2, shuffle=True, epochs=5)

model.save("model.h5")
