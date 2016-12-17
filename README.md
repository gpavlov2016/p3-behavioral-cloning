Network architecture:
=====================
The network is build in two parts - the bottom layers are based on VGG16 network 
and the top 3 layers are fully connected layers with output size of 1024 each
with relu activation function and varying probabilities of dropout between them.
The droupout probabilities are 0.5, 0.2 and 0.1 between fc1-fc2-fc3 (bottom top).

Training approach:
==================
This network is of a considerable size and combined with the large size of the input
it takes a lot of time to train. To cope with this problem I loaded weights for
convolutional layers of the model pretrained on imagenet dataset and froze them.
In the first stage I trained the top fully connected layers only on the dataset
for about 100 epochs and when the weights converged I unfroze the convolutional 
layers and started a second phase of training on the whole network for additional
100 epochs. This approach prevented from convolutional weights to diverge due to 
unstable backprop from untrained top layers. The assumption here is that some of
the low level features are similar between the two datasets (imagenet and car simulator)
and their values can be reused in this case. I also saved the weights between the 
first and second phases and checked performance using the simulator on both of them. 
The weights after the second phase performed better.
The approach described above produced a model with a loss of 0.03 and is capable 
of driving for almost two rounds on the first track. To improve the results I 
retrained the model with the weights on udacity data for 20 more epochs, now
using all 3 camera images. The turn angle (Y label) was adjusted for left camera
by 0.1 and for left camera by -0.1. This reduced the loss to 0.0087 on Udacity
data (which is smoother so smaller loss is expected) and fortunetely also resulted
in better driving, allowing the car to drive around for at least 5 laps.
The surprizing part was how well this model handled difficult turns but had 
a difficulty on relatively straing portion of the track. Probably can be improved
by additional training on smooth data like Udacity's dataset.

Samples:
========
Initially I started with about 4000 samples, dividing them to 25% test set,
18% validation set and the rest training sets. However this was not enough for 
very good permormance, though one of the training weights created over 200+200
epochs allowed the car to drive by itself for a loop and a half.
With this in mind I recorded more data reaching 14K samples dividing them to 
10% test, 9% validation and almost 80% training sets. The increase in the amount
of data also required working with a batch generator because the whole data 
didn't fit to machine's memory. There were two multithreaded nested generators 
supporting train, val and test sets. More on the generator design see below.

Batch generators:
=================
The outer bach generator called threaded_generator launches the inner batch
generator called batch_generator in a separate thread and caches 10 outputs
of the latter. Each output consists of one batch (around 128 samples).
The inner generator supports three types of data - train, test and val. In 
the aftermath the division to test and val is redundant since the results 
on both were pretty similar so it would have worked with only train and test
data.
To support three data types the batch generator accepts besides the batch size
a second parameter that selects the type. Based on this parameter one of three
csv file arrays are chosen. The arrays are prepared earlier in the data loading
phase where all the csv files are read (multiple data directories are supported)
and the rows are merged in one array. Then the array is split into parts train
test and val and assigned to different variable. This approach simplifies data
shuffling since the csv rows contain both features and labels and are small in 
size.
After the batch generator decides on the appropriate csv rows array it randomly
samples batch_size rows from the array, reads the images from respective files,
resized them, normalizes the data and appends them to images array (X). The labels
are appended to labels array and if three cameras are used the labels for left
and right cameras are adjusted by 0.1 and -0.1 respectively.

Data preprocessing:
===================
There are two main steps to data preprocessing:
1. Resizing - from the (320, 160) original size to (80, 40) by using OpenCV. The 
image is not grayified in an attempt to help the model to recognize borders which
mostly have distinct red or yellow colors.
2. Normalization - adjusting the data to the range of -0.5 - 0.5 with a mean of 0
to improve computational stability, conviniently done by the preprocess_data function
imported from keras.models.VGG16

Data generation:
================
In a hindsight this was more difficult and more important that I have initialy assumed.
I recorded 5 datasets each with about three laps but the quality of driving wasn't
great since I used keyboard for steering, and in the worst cases the car went out of the
track. This created a problem because in some experiments I noticed that the car was
crashing in the same place when I did bad steering job when recording data, meaning
that I was teaching the car bad habits :). This was also a good sign though because
I can clearly see that the model is learning and realizing the "behavior cloning"
concept. At some stage I was able to create a model that drove for almost two rounds
by training on two datasets but then the progress stalled. One of the fallback options
was to manually delete pictures and lines of csv that exhibited bad driving decisions
but this is very time consuming and not accurate process. Fortenutely Udacity released
its own data which didn't have very bad steering decisions and after training on this 
data the car was able to drive for at least 5 laps.

Experiments and results:
========================
Here are some of the experiments I did but did not graduate to the final solution.
1. Several pretrained archs such as Inception and ResNet, the VGG arch performed
slightly better then them in short experiments.
2. Combining speed to the decision. I didn't observe significant improvement adding
speed to the features and in some cases the performance got worse. It's hard at this
point to isolate whether it was the speed or defficiency of other parameters.
3. Training small models ranging from 16x32x64 conv layers to 50x150x250, different
sizes of FC layers (4096x4096x4096 - 1024x512x256), the top bottom FC layer of 1024
seems like the sweet spot thought the results are not conclusive.
4. Different dropout probabilities from 0 to 0.5 with piramide pattern (larger
dropouts for larger layers). Higher dropouts required longer training but did not
produce better results.


