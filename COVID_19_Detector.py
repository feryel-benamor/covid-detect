import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

TRAIN_PATH = "D:/3_DSEN_B/projet_bigdata/covid_radiography/train"
VAL_PATH = "D:/3_DSEN_B/projet_bigdata/covid_radiography/Test"

#CNN Based Model in Keras

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()


train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'D:/3_DSEN_B/projet_bigdata/covid_radiography/train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

train_generator.class_indices

validation_generator = test_datagen.flow_from_directory(
    'D:/3_DSEN_B/projet_bigdata/covid_radiography/Test',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

hist = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = 8,
    epochs = 40,
    validation_data = validation_generator,
    validation_steps = 2
)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('keras.h5')

