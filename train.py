from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt
import os

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset/train'
test_path = 'dataset/val'

# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

transfer = ResNet50V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in transfer.layers:
    layer.trainable = False

# folders = glob('C:/Users/Raghu/Desktop/CNN/Tomato Leaf Disease/Dataset/train/*')

x = Flatten()(transfer.output)

prediction = Dense(20, activation='softmax')(x)

# create a model object
model = Model(inputs=transfer.input, outputs=prediction)


model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/val',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')
r=model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=15,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set)
        )

model.save('food-resnet-50-v2-updated-epoch_15.h5')


plt.plot(r.history['accuracy'],label='train accuracy')
plt.plot(r.history['val_accuracy'],label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('Accuracy_resnet-50-v2-updated-epoch_15')
# summarize history for loss
plt.plot(r.history['loss'],label="train loss")
plt.plot(r.history['val_loss'],label="validation loss")
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('Loss_resnet-50-v2-updated-epoch_15')