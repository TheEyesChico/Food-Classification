import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import operator

path='food-resnet-50-v2-updated-epoch_15.h5'
model = load_model(path)

paths=[]
folder_path=r'C:\Users\Raghu\Desktop\CNN\Food Classification\uploads'
# real=os.path.basename(folder_path)

for i in os.listdir(folder_path):
    paths.append(os.path.join(folder_path,i))

print("Folder has {} items".format(len(paths)))

# test_images=os.listdir(r'C:\Users\Raghu\Desktop\CNN\Food Classification\dataset\test')
# num_classes=[x for x in range(0,len(test_images))]
# categories=dict(zip(test_images,num_classes))
# print(categories)

def model_predict(images, model,categories,real):
    z=0
    orignal_class=categories[real]
    for i in images:
        img = image.load_img(i, target_size=(224, 224))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        ## Scaling
        x = x / 255
        x = np.expand_dims(x, axis=0)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        # x = preprocess_input(x)

        preds = model.predict(x)

        preds = np.argmax(preds, axis=1)

        if preds[0]==orignal_class:
            z=z+1

        print("Image - {0},\t Prediction - {1}".format(os.path.basename(i),preds[0]))

    accuracy=(z/len(images))*100

    return("Accuracy : %f"%accuracy)

def model_predict_random(images, model):
    category={0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani', 6: 'dhokla',
     7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls', 11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa',
     14: 'momos', 15: 'paani_puri', 16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'}
    for i in images:
        img = image.load_img(i, target_size=(224, 224))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        ## Scaling
        x = x / 255
        x = np.expand_dims(x, axis=0)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        # x = preprocess_input(x)

        preds = model.predict(x)

        preds = np.argmax(preds, axis=1)

        print("Image - {0},\t Prediction - {1} ( {2} )".format(os.path.basename(i),preds[0],category.get(preds[0])))

    return None

# a=model_predict(paths,model,categories,real)
# print(a)

b=model_predict_random(paths,model)
print(b)