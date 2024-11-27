import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Image size
sz = 128

# Step 1 - Building the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units= 3, activation='softmax'))  # For 27 classes (A-Z and space)

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    'data2/train',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical',
)

test_set = test_datagen.flow_from_directory(
    'data2/test',
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical',
)

# Training the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=5,
    validation_data=test_set,
    validation_steps=len(test_set),
)

# Saving the model architecture
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)

# Saving the weights
classifier.save_weights("model-bw.weights.h5")
