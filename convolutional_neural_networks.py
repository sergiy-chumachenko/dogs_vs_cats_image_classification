# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#%%
# Part 1 - Building The Convolutional Neural Network (CNN)
# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#%%
# Initialize the CNN
classifier = Sequential()

#%%
# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

#%%
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#%%
# Add second Convolution Layer
classifier.add(Convolution2D(32, (3, 3),  activation='relu'))

#%%
# Add second Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#%%
# Step 3 - Flattening
classifier.add(Flatten())

#%%
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#%%
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%
# Part 2 - Fitting the CNN to the images
# We are using ImageDataGenerator to prevent overfitting: 'https://keras.io/preprocessing/image/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#%%
training_set = train_datagen.flow_from_directory(
        '/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#%%
classifier.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=3,
        validation_data=test_set,
        validation_steps=2000)
