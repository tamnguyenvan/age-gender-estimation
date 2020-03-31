"""Train age and gender estimation.
"""
import os
import pickle as pkl
import numpy as np
import tensorflow as tf
from resnet import resnet18_v2
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import cv2
import utils
tf.enable_eager_execution()

# Fix RTX card
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

IMAGE_SIZE = 112
NUM_TRAIN = 188991
NUM_VAL = 20999
AGE_CLASSES = 101
GENDER_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 50


def lr_schedule(epoch):
    """Learning rate schedule callback."""
    lr = 1e-3
    if epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def parse_fn(path, age, gender):
    """Parse image function."""
    image = tf.io.decode_jpeg(tf.io.read_file(path, 'rb'), channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = image / 255.
    age = tf.one_hot(age, AGE_CLASSES)
    gender = tf.one_hot(gender, GENDER_CLASSES)
    return image, (age, gender)


@tf.function
def mae(y_true, y_pred):
    classes = tf.range(AGE_CLASSES, dtype=tf.float32)
    age_true = tf.reduce_sum(y_true * classes)
    age_pred = tf.reduce_sum(y_pred * classes)
    return tf.reduce_mean(tf.abs(age_true - age_pred))


# Data preparation
data_dir = 'data'
(train_paths, train_ages, train_genders), (val_paths, val_ages, val_genders) = \
        utils.load_data(data_dir, 'imdb,wiki')

print('Number of training examples:', len(train_paths))
print('Number of validation examples:', len(val_paths))

train_data = tf.data.Dataset.from_tensor_slices((train_paths, train_ages, train_genders))
train_data = train_data.shuffle(1000) \
        .map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .repeat() \
        .prefetch(tf.data.experimental.AUTOTUNE)

val_data = tf.data.Dataset.from_tensor_slices((val_paths, val_ages, val_genders))
val_data = val_data.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

# Build the model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = resnet18_v2(input_shape, include_top=False)
age_out = layers.Dense(units=AGE_CLASSES,
                       activation='softmax',
                       name='age')(base_model.output)
gender_out = layers.Dense(units=GENDER_CLASSES,
                          activation='softmax',
                          name='gender')(base_model.output)
model = Model(base_model.input, [age_out, gender_out])
model.count_params()
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss={'age': 'categorical_crossentropy',
                    'gender': 'categorical_crossentropy'},
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics={'age': 'accuracy', 'gender': 'accuracy'})

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'age_gender_resnet_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=3,
                               min_lr=0.5e-6)
early_stopping = EarlyStopping(monitor='val_loss',
                               mode='auto',
                               patience=5,
                               verbose=1,
                               restore_best_weights=True)
callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

hist = model.fit(
    train_data,
    steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
    validation_data=val_data,
    validation_steps=NUM_VAL // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks)

np.save('history.npy', hist)
