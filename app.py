import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def get_model():
    global model
    model = load_model('model_SmartBin.h5')
    print(" * Loading Keras model...")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def ToClass(image):
    if image.shape[-1] > 1:
        return image.argmax(axis=-1)
    else:
        return (image > 0.5).astype('int32')
def ResultClass(R):
    if(ToClass(R)==0):
        return("can't recycles")
    else:
        return("recycles")

get_model()

@app.route("/")
def render():
    return render_template("app.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == "GET":
        return jsonify({"respone":"Get Request Called"})

    elif request.method == "POST":
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)

        return jsonify(ResultClass(prediction))

if __name__ == "__main__":
    app.run(debug=True, port=5000)