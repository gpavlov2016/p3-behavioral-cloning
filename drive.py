import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
target_shape = (80, 40) #original size 320, 160

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = np.array([float(data["speed"])])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = cv2.resize(image_array, target_shape)
    transformed_image_array = image_array[None, :, :, :]
    transformed_image_array = preprocess_input(transformed_image_array.astype('float'))
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #print('predicting')
    steering_angle = float(loaded_model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    #args = parser.parse_args()

    model_file = 'model.json'
    with open(model_file, 'r') as jfile:
        loaded_model = model_from_json(json.load(jfile))
    loaded_model.summary()
    weights_file = model_file.replace('json', 'h5')
    loaded_model.load_weights(weights_file)

    loaded_model.compile("adam", "mse")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
