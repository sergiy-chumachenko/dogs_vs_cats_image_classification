# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building The Convolutional Neural Network (CNN)
# Importing the Keras libraries and packages
#%%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
#%%
classifier = Sequential()

# Step 1 - Convolution
#%%
classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
#%%
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
#%%
classifier.add(Flatten())

# Step 4 - Full connection
#%%
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
#%%
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
