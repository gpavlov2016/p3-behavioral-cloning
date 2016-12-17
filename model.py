import csv
import numpy as np
import cv2

target_shape = (80, 40)  # original size 320, 160
nb_cams = 3
angle_shift = 0.1


def load_csv(directory=''):
    # data_dirs = ['dataset-1', 'dataset-2', 'dataset-3', 'dataset-4', 'dataset-5', 'dataset-water', 'data-udacity']
    data_dirs = ['dataset-1', 'dataset-2']

    global csv_rows_train, csv_rows_val, csv_rows_test
    csv_rows = []

    for dr in data_dirs:
        log_filename = dr + '/driving_log.csv'
        img_path = dr + '/IMG/'

        with open(log_filename, 'r') as csvfile:
            logreader = csv.reader(csvfile)
            if dr == 'data-udacity':  # udacity data format
                row = next(logreader)
                print(row)
            for row in logreader:
                if dr == 'data-udacity':  # udacity data format
                    row[0] = dr + '/' + row[0].strip()  # center
                    row[1] = dr + '/' + row[1].strip()  # left
                    row[2] = dr + '/' + row[2].strip()  # right
                else:
                    row[0] = img_path + row[0].split('\\')[-1]  # center
                    row[1] = img_path + row[1].split('\\')[-1]  # left
                    row[2] = img_path + row[2].split('\\')[-1]  # right
                csv_rows.append(row)
    csv_rows = np.array(csv_rows)

    # split the rows to train test val
    from sklearn.model_selection import train_test_split
    csv_rows_train, csv_rows_test = train_test_split(csv_rows, test_size=0.1)
    csv_rows_train, csv_rows_val = train_test_split(csv_rows_train, test_size=0.1)

    print('CSV loaded, #train = ', len(csv_rows_train),
          '#val = ', len(csv_rows_val),
          '#test = ', len(csv_rows_test))


def load_image(img_filename, angle, images, angles):
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    angles.append(float(angle))


def batch_generator(batch_size, source='train'):
    if source == 'train':
        csv_rows = csv_rows_train
    elif source == 'val':
        csv_rows = csv_rows_val
    elif source == 'test':
        csv_rows = csv_rows_test
    else:
        print('Error data segment unknown = ', source)

    row_indices = range(csv_rows.shape[0])

    while 1:
        chosen_indices = np.random.choice(row_indices, size=int(batch_size / nb_cams))
        # print(chosen_indices)
        chosen_rows = csv_rows[chosen_indices]
        # print(chosen_rows.shape)
        images = []
        angles = []
        for row in chosen_rows:
            load_image(row[0], float(row[3]), images, angles)  # center
            if nb_cams == 3:
                load_image(row[1], float(row[3]) + angle_shift, images, angles)  # left
                load_image(row[2], float(row[3]) - angle_shift, images, angles)  # right

        preprocessed_imgs = preprocess_input(np.array(images).astype('float'))
        X_train = preprocessed_imgs
        y_train = np.array(angles)
        # print('Batch generated.')
        yield (X_train, y_train)


# from https://github.com/Lasagne/Lasagne/issues/12
def threaded_generator(generator, num_cached=10):
    import queue
    queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def show_predictions(X_test, Y_test):
    print('Predicting on test set...')
    predictions = model.predict(X_test)
    sum_p = 0
    sum_y = 0
    for i in range(32):
        sum_p += predictions[i][0]
        sum_y += Y_test[i]
        print('p: %.4f' % predictions[i][0], ', y: %.4f' % Y_test[i],
              'c_p: %.4f' % sum_p, ', c_y: %.4f' % sum_y)


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1)(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
my_adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=my_adam)
# model.compile(optimizer='rmsprop', loss='mean_absolute_error')

model.summary()

load_csv()

batch_size = nb_cams * int(128 / nb_cams)  # must be multiply of nb_cams
samples_per_epoch = batch_size * int(len(csv_rows_train) / batch_size)
nb_epoch = 100
nb_val_samples = 5 * batch_size

history = model.fit_generator(threaded_generator(batch_generator(batch_size, 'train')),
                              samples_per_epoch, nb_epoch=nb_epoch,
                              verbose=1, validation_data=batch_generator(batch_size, 'val'),
                              nb_val_samples=nb_val_samples)

print('Training completed.')

# Visualize training history
def visualize_history():
    import matplotlib.pyplot as plt
    import numpy

    print(history.history.keys())
    #%matplotlib inline

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0,np.max(history.history['val_loss'])])
    plt.show()

#visualize_history()

def eval_and_save():
    test_samples = nb_cams*int(1000/nb_cams)
    gen = threaded_generator(batch_generator(test_samples, 'test'))
    X_test, Y_test = next(gen)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score (MSE): ', score)

    filepath = 'vgg16-pretrained-gen-' + '%.4f' % score
    # save model architecture:
    json_string = model.to_json()
    import json
    with open(filepath + '.json', 'w') as outfile:
        json.dump(json_string, outfile)
    print('Model saved to ', filepath + '.json')
    # save weights:
    model.save_weights(filepath + '.h5')
    print('Weights saved to ', filepath + '.h5')

eval_and_save()

#Train the whole model:
for layer in model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
my_adam = Adam(lr=0.00001)
model.compile(loss='mse', optimizer=my_adam)

#nb_epoch = 10
print('batch_size: ', batch_size)
print('samples_per_epoch: ', samples_per_epoch)
print('nb_epoch: ', nb_epoch)
print('nb_val_samples: ', nb_val_samples)

history = model.fit_generator(threaded_generator(batch_generator(batch_size, 'train')),
                    samples_per_epoch, nb_epoch=nb_epoch,
                    verbose=1, validation_data=batch_generator(batch_size, 'val'),
                    nb_val_samples=nb_val_samples)
eval_and_save()

