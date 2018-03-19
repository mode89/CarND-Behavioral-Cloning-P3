import csv
import cv2
import numpy
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

def load_log(path, images, steeringAngles):
    with open(path) as drivingLog:
        reader = csv.reader(drivingLog)
        for line in reader:
            imagePath = line[0]
            image = cv2.imread(imagePath)
            images.append(image)
            steeringAngles.append(float(line[3]))

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
model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(trainX, trainY, validation_split=0.2, shuffle=True, epochs=11)

model.save("model.h5")
