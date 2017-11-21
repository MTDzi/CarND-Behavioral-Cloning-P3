# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in `keras` that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./model_summary.png "Output from `model.summary()`"
[image1]: ./center.jpg "Center Image"
[image2]: ./recovery_1.jpg "Recovery Image 1"
[image3]: ./recovery_2.jpg "Recovery Image 2"
[image4]: ./recovery_3.jpg "Recovery Image 3"
[image5]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the `drive.py` file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The `model.py` file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed
My model is an extension of the LeNet model, I call it "LeNet like model". The model is produced by the `get_lenet_like_model` function (lines 77-110).
The model starts with a 1x1 convolution (with 4 filters), which amounts to a feature map consisting of 4 "maps", each being a linear combination of the input RGB image.

Next, there are three consecutive (5x5 convolution + max pooling + ELU activation) modules, similar to what can be found in the actual LeNet model. The number of filters in these modules are: 16, 32, and 16.

These modules are followed by a flatten operation, and dropout (all dropout operations had a `keep_prob` of 0.5), next followed be a fully connected (FC) layer with 256 neurons, followed by an FC layer of 128 neurons, followed be an FC layer of 64 neurons, and finalized be a single regression neuron at the end.

Here is a visualization of the architecture:

![alt text][image0]

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (`model.py` lines 98, 101, and 104), but also I used weight decay in all layers of the net (with a coefficient of *1e-3*).

The model was trained and validated on disjoint parts of the whole data sets to ensure that the model was not overfitting (code line 25 in `read_data.py`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
I used an ADAM optimizer, so the learning rate was not tuned manually (`model.py` line 25), but with an initial learning rate of *1e-4*.

#### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In particular, because I struggled A LOT with the left turn right after the bridge, I took that turn numerous (20+) times.

For details about how I created the training data, see the next section.


### Model Architecture and Training Strategy
#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to start with the standard LeNet architecture, then increase the complexity (to get a very low MSE on the training data, approximately *0.002*, even at the cost of high MSE on validation), and then add regularization to get a low error on the validation set.
I've started with dropout, experimented with BatchNorm (but it didn't improve the validation error), and used weight decay throughout the network.

I've also experimented with other architectures (all defined in the `model.py` script), most notably, with the NVIDIA architecture. I was able to get comparable, and/or better results (depending on the set up) on the validation set, but not in terms of how the model drove the car in autonomous mode.

To combat overfitting, I used various additions to the model (already mentioned above), but also, I randomly flipped the images horizontally (with a probability of *0.25*).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track -- in particular the first turn after the bridge. To improve the driving behavior in these cases, I recorded taking that turn many times, but what eventually proved most useful was the 1x1 convolution transforming the RGB image into a 4-dimensional feature map. This way the model was able to identify the muddy track as a "no-go" (at least that's my interpretation).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from critical situations. These images show what a recovery looks like starting from:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I did NOT collect data on track two, I wanted to stick to track 1, and perhaps attack track 2 on some other occasion.

To augment the data sat, I also flipped images horizontally, thinking that this would allow the model to better generalize, rather than identify artifacts pertinent only to certain turns. For example, here is an image that has been flipped:

![alt text][image1]
![alt text][image5]

After the collection process, I had 3 x 60.000 = 180.000 of data points (the "3 x" comes from the fact that for each position of the car I took images from: center, left, and right).  I also cropped 50 rows of the image array from the top, which resulted in a tensor of shape: (110, 320, 3). I then
Finally,

I finally randomly shuffled the data set and put 1% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was (over/under)fitting. The ideal number of epochs was 10, I used an ADAM optimizer so that manually training the learning rate wasn't necessary.
