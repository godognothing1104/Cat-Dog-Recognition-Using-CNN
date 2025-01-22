#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
K.clear_session()
tf.keras.backend.clear_session()
#%% md
# Data Preprocessing
#%%
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

#%%
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
#%% md
# Build the CNN
#%%
cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#%% md
# Pooling
#%%
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#%% md
# Second convolutional layer
#%%
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#%% md
# Flattening
#%%
cnn.add(tf.keras.layers.Flatten())
#%% md
# Full connectioon
#%%
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#%% md
# Output layer
#%%
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#%% md
# Train the CNN
#%%
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x = training_set, validation_data= test_set,epochs=25 )
#%% md
# Prediction
#%%
import numpy as np #expect predicting dog
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
#%%
print(prediction)