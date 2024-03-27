from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras import models
import cv2
import os
import numpy as np
from math import floor

# app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/temp'
model = models.load_model('model/plant_disease_classification.h5')

# routes
@app.get("/")
def home():
    return render_template("pages/home.html")

@app.get("/data")
def data():
    return render_template("pages/data.html")

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():    
    if request.method == 'POST' and request.files['image']:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prediction=make_prediction(file_on_disk)
        os.remove(file_on_disk)
        return render_template("pages/prediction.html", prediction=prediction)
        
    return render_template("pages/prediction.html")

# model interaction
def make_prediction(image):
    categories = os.listdir("dataset/train")

    img = cv2.imread(image)
    img = cv2.resize(img, (128,128))
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img=np.expand_dims(img, axis=0)

    result = model.predict(x=img)

    for i in range(len(categories)):
        if result[0][i] == np.max(result[0]):
            return [(categories[i]).replace("___", " ").replace("_", " "), str(floor(result[0][i] * 100))]

if __name__ == "__main__":
    app.run()