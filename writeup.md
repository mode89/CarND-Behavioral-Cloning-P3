**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[center_lane_driving]: ./examples/center.jpg "Center Lane Driving"
[left_camera_image]: ./examples/left.jpg "Left Camera Image"
[right_camera_image]: ./examples/right.jpg "Right Camera Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* wirteup.md or summarizing the results
* video.mp4 showing driving in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution
neural network. The file shows the pipeline I used for training and
validating the model. The names of functions and variables are
self-explanatory and making the code easily readable. The code uses python
generators rather than storing the data inside memory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've borrowed the model's architecture from the Nvidia's paper End-to-End
Learning for Self-Driving Cars. The model consists of five convolutional
layers with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is
normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that
the model was not overfitting. The model was tested by running it through
the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned
manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I gathered
the training data by driving the car in manual mode in the center of the
lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a single layer regression neural network. I created
a network with a single flatten layer and a single fully-connected layer.
After 8 training epochs the training loss reached ~2000 and the validation
loss reached ~2500. Driving in autonomous mode was very rough - steering
angle was changing rapidly. I used this model to test the training process
and autonomous driving mode.

Then I have implemented a model with architecture borrowed from the Nvidia's
paper End-to-End Learning for Self-Driving Cars. I thought this model might
be appropriate, because in the paper they showed that their model is able to
learn the task of lane following without manual segmentation abstraction and
path planning.

In order to gauge how well the model was working, I split my image and
steering angle data into a training and validation set. I found that my
first model had a low mean squared error on the training set but a high mean
squared error on the validation set. This implied that the model was
overfitting.

To combat the overfitting, I modified the model by putting dropout layers
before each of the fully connected layers of the original model. It improved
the validation loss and made it smaller than the training loss.

Using Lambda class, I've embedded a cropping layer, scaling layer,
grayscaling layer, color normalization layer. It helped to reduce the size
of the model, speed up calculations and make model generalize better.

I used python generators to avoid extensive memory consumption. But it
reduced the speed of training.

Used ModelCheckpoint callback to collect and test in simulator intermediate
versions of model. It helped to identify the maximum value of validation
loss that should be suitable for autonomous driving.

Used EarlyStopping callback helped to reach better loss values without
spending too much time on training.

The final step was to run the simulator to see how well the car was driving
around track one. There were a few spots where the vehicle fell off
the track. To improve the driving behavior in these cases, I used the
mentioned above tricks improving the validation loss and then I captured
more training data for the weak spots.

At the end of the process, the vehicle is able to drive autonomously around
the track without leaving the road. I left it running overnight, and the car
finished more then a hundred laps without leaving the track.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with
the following layers:

| Layer             | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| Crop              | Cut out top 70 lines and bottom 25 lines                              |
| Scale Down        | Down to 64 x 128                                                      |
| Grayscale         |                                                                       |
| Normalize         |                                                                       |
| Convolution       | Kernel 5 x 5, strides 2 x 2, depth 24, valid padding, ReLU activation |
| Convolution       | Kernel 5 x 5, strides 2 x 2, depth 36, valid padding, ReLU activation |
| Convolution       | Kernel 5 x 5, strides 2 x 2, depth 48, valid padding, ReLU activation |
| Convolution       | Kernel 3 x 3, strides 1 x 1, depth 64, valid padding, ReLU activation |
| Convolution       | Kernel 3 x 3, strides 1 x 1, depth 64, valid padding, ReLU activation |
| Flatten           |                                                                       |
| Dropout           | Rate 0.5                                                              |
| Fully-connected   | 1164 outputs                                                          |
| Dropout           | Rate 0.5                                                              |
| Fully-connected   | 100 outputs                                                           |
| Dropout           | Rate 0.5                                                              |
| Fully-connected   | 50 outputs                                                            |
| Dropout           | Rate 0.5                                                              |
| Fully-connected   | 10 outputs                                                            |
| Dropout           | Rate 0.5                                                              |
| Fully-connected   | 1 output                                                              |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one,
using center lane driving. First lap I drove in the clockwise direction, and
the second lap I drove in the counter clockwise direction. Here is an
example image of center lane driving:

![Center Lane Driving][center_lane_driving]

I used images captured by the left and the right cameras, so that the
vehicle would learn recovering from the side of the track. Here are the
examples of the images captured by the left and the right cameras:

![Left Camera Image][left_camera_image]
![Right Camera Image][right_camera_image]

I labeled the images captured by the left camera with a steering angle
higher than the steering angle of the center camera images by value of 0.2.
I labeled the images captured by the right camera with a steering angle
lower than the steering angle of the center camera images by value of 0.2.

I finally randomly shuffled the data set and put 20% of the data into a
validation set. I used this training data for training the model. The
validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 40. I used an adam optimizer so that
manually training the learning rate wasn't necessary.

There were two hard spots - the second turn and the third turn (in the order
of appearance in autonomous mode). I ran over the second turn 30 more times,
and 20 more times over the third turn to gather enough training data to pass
them in the autonomous mode.
