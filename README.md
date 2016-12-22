## Changes from previous submission
1. Simplified architecture to 3 convolutional layers followed by 3 fully connected.
2. Image preprocessing - converted to VHS and used the S channel, modified normalization.

## Network architecture
The network consists of three convolutional layers increasing in depth and 3 fully connected
layers decreasing in size. Dropout is employed between the fully connected layers and 
activation function is relu.
Here is the network architecture as shown by keras model.summary():
`
___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 40, 80, 32)    320         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 20, 40, 32)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 40, 64)    18496       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 10, 20, 64)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 20, 128)   73856       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 10, 128)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6400)          0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           3200500     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 500)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            1010        dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
====================================================================================================
Total params: 3344293
____________________________________________________________________________________________________
`
## Training approach:
As opposed to the large network I had in previous submission, this one is much smaller
with only 3M parameters, also helped the fact that only 1 channel instead of three were
used. The smaller size of the network allowed to make several experiments with different
datasets and surprizingly the first one I tried, which is the Udacity dataset produced
very good results after 20 epochs that took only few minutes to run. On the other hand,
the training on data I recorded, even together with Udacity data produce inferior results
in which the car crashes in the first lap.

## Data:
The data consists of 8036 samples which were divided to 6508 train samples, 724 validation
samples and 804 test samples.

## Batch generators:
There are two multithreaded nested generators supporting train, val and test sets. The 
outer bach generator called threaded_generator launches the inner batch generator 
called batch_generator in a separate thread and caches 10 outputs of the latter. Each 
output consists of one batch (around 128 samples). The inner generator supports three 
types of data - train, test and val. In the aftermath the division to test and val is 
redundant since the results on both were pretty similar so it would have worked with 
only train and test data.
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
preprocesses the images and appends them to images array (X). The labels are appended 
to labels array and if three cameras are used the labels for left and right cameras 
are adjusted by 0.1 and -0.1 respectively.

## Data preprocessing:
There are three main steps to data preprocessing:
- Resizing - from the (320, 160) original size to (80, 40) using OpenCV resize method.
- Color space conversion - the image is converted to HVS format and only the S channel
   is used. 
- Normalization - scaling the data to the range of 0-1

## Data generation:
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

## Experiments and results:
Here are some of the experiments I did but did not graduate to the final solution.
- Several pretrained archs such as Inception and ResNet, the VGG arch performed 
slightly better then them in short experiments.
- Combining speed to the decision. I didn't observe significant improvement adding
speed to the features and in some cases the performance got worse. It's hard at this
point to isolate whether it was the speed or defficiency of other parameters.
- Training small models ranging from 16x32x64 conv layers to 50x150x250, different
sizes of FC layers (4096x4096x4096 - 1024x512x256), the top bottom FC layer of 1024
seems like the sweet spot thought the results are not conclusive.
- Different dropout probabilities from 0 to 0.5 with piramide pattern (larger
dropouts for larger layers). Higher dropouts required longer training but did not
produce better results.


