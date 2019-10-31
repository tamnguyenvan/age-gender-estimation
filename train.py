import os
import pickle as pkl
import numpy as np
import keras
import tensorflow as tf
from datetime import datetime
from keras import backend as K
from dataset_utils import load_data, DataGenerator

# fix rtx gpu bug
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
K.set_session(tf.Session(config=config))


def build_model(input_shape, learning_rate):
    base_model = keras.applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    pred_age = keras.layers.Dense(101, activation='softmax', kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(5e-4), name='age')(x)
    pred_gender = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(5e-4), name='gender')(x)
    model = keras.models.Model(base_model.input, [pred_age, pred_gender])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={'age': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
        metrics=['acc']
    )
    return model


# parameters
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-2
INPUT_DIM = (64, 64)
INPUT_SHAPE = (*INPUT_DIM, 3)

DATA_DIR = 'data'
DB = 'imdb,wiki'
CHECKPOINT_DIR = 'saved_models'
LOG_DIR = 'logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')

# load data
(train_paths, train_ages, train_genders), (val_paths, val_ages, val_genders) = load_data(DATA_DIR, DB)
print('Train/Val: {}/{}'.format(len(train_paths), len(val_paths)))

num_train = len(train_paths)
num_val = len(val_paths)
train_data = DataGenerator(train_paths, train_ages, train_genders, INPUT_DIM, BATCH_SIZE, dtype='float32')
val_data = DataGenerator(val_paths, val_ages, val_genders, INPUT_DIM, BATCH_SIZE, dtype='float32')

# build the model
model = build_model(INPUT_SHAPE, LEARNING_RATE)
model.summary()

# setup callbacks
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'age_gender_model.{epoch:02d}-{val_loss:.2f}.hdf5')
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=2, verbose=1, min_lr=0.5e-6, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

hist = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[lr_reducer, checkpoint, tensorboard],
    workers=6
)

history_file = os.path.join(CHECKPOINT_DIR, 'history.pkl')
pkl.dump(open(history_file, 'wb'), hist.history)