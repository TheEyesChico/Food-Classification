from flask import Flask, render_template, request, redirect
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH ='food-resnet-50-v2-updated-epoch_15.h5'
model = load_model(MODEL_PATH)

def model_predict(image_path, model):
    print(image_path)
    category={0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani', 6: 'dhokla',
     7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls', 11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa',
     14: 'momos', 15: 'paani_puri', 16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'}

    img = image.load_img(image_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)

    return "The food item is {}".format(category.get(pred[0]))

app.config["IMAGE_UPLOADS"] = "C:/Users/Raghu/Desktop/CNN/Food Classification/static/uploads/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def upload_image():

    if request.method == 'POST':

        if request.files:

            f=request.files["file"]

            if f.filename == "":
                print("No filename")
                return redirect(request.url)

            if not allowed_image(f.filename):
                print("Image extension is not a picture")
                return redirect(request.url)

            else:
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'uploads', secure_filename(f.filename))
                f.save(file_path)
                # image.save(os.path.join(app.config["IMAGE_UPLOADS"], secure_filename(image.filename)))
                # print("Image Saved")
                # save_path=os.path.join(app.config["IMAGE_UPLOADS"], secure_filename(image.filename))
                # print(save_path)
                # print(type(save_path))
                preds=model_predict(file_path,model)
                result=preds
                return result

        return redirect(request.url)
        
    return render_template("upload_image.html")


if __name__=="__main__":
    app.run(debug=True)