
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np

path='food-inceptionv3-epoch_15.h5'
model = load_model(path)

img=image.load_img('C:/Users/Raghu/Desktop/293.jpg',target_size=(224,224))
x=image.img_to_array(img)
print(x)
print(x.shape)
x=x/255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
print(img_data.shape)

model.predict(img_data)
a=np.argmax(model.predict(img_data), axis=1)
print(a)

#
#
# def model_predict(img_path, model):
#     print(img_path)
#     img = image.load_img(img_path, target_size=(224, 224))
#
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     ## Scaling
#     x = x / 255
#     x = np.expand_dims(x, axis=0)
#
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     # x = preprocess_input(x)
#
#     preds = model.predict(x)
#     print(preds)
#     preds = np.argmax(preds, axis=1)
#     print(preds)
#     preds=preds+1
#
#     if preds == 0:
#         preds = "Tomato Bacterial Spot"
#     elif preds == 1:
#         preds = "Tomato Early Blight"
#     elif preds == 2:
#         preds = "Tomato Healthy"
#     elif preds == 3:
#         preds = "Tomato Late Blight"
#     elif preds == 4:
#         preds = "Tomato Leaf Mold"
#     elif preds == 5:
#         preds = "Tomato Septoria leaf spot"
#     elif preds == 6:
#         preds = "Tomato Spider mites Two spotted spider mite"
#     elif preds == 7:
#         preds = "Tomato Target Spot"
#     elif preds == 8:
#         preds = "Tomato Tomato mosaic virus"
#     elif preds == 9:
#         preds = "Tomato Yellow Leaf Curl Virus"
#
#     return preds
#
# a=model_predict('C:/Users/Raghu/Desktop/53.jpg', model)
# print(a)