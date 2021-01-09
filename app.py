import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2
import imutils

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
# from keras.models import load_model
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# You can load your model or covid.h5/pneu-3.h5 in modelsc
MODEL_PATH = 'models/pneu-3.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary

print('Model loading...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.vgg16 import VGG16
# model = VGG16(weights='imagenet', include_top=False)
#graph = tf.get_default_graph()

print('Model loaded. Started serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('Begin Model Prediction...')

        # Make prediction
        image = cv2.imread(file_path)
        image = image.copy()
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        (normal, pneumonia) = model.predict(image)[0]


        label = "Pneumonia" if pneumonia > normal else "normal"
        proba = pneumonia if pneumonia > normal else normal
        label = "The X-Ray Analyzed is {} - {:.2f}%".format(label, proba * 100)

        print('End Model Prediction...')
        os.remove(file_path)

        return label
    return None

if __name__ == '__main__':
    # app.run(debug=False, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
