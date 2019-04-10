# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 04:31:04 2019

@author: Jeffrey
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 01:56:43 2019

@author: jeffr
"""

#Part 1 - Building the CNN

#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu'))
 
#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images gained from KERAS documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'cellimages/Par',
        target_size=(64, 64), 
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=(8000/32),
        epochs=25,
        validation_data= test_set,
        validation_steps=(2000/32))

#Part 3 - Testing CNN with my own dog "Lincoln!"
import numpy as np
from keras.preprocessing import image
from PIL import Image
test_image = image.load_img('dataset\my_dog\lincoln.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_set.class_indices
if result == 1:
    print("Lincoln is a Dog")
else:
    print("Lincoln is a Cat")
    img = Image.open('dataset\my_dog\lincoln.jpg')
    img.show()