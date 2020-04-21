"""Fine tunning the model on APPA-real training dataset."""
import os
import argparse
from sys import argv

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

import utils
import augmentor
import models

# tf.enable_eager_execution()
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Fix tensorflow bug on rtx card
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)


def parse_args(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory.')
    parser.add_argument('--dataset', type=str, default='imdb,wiki,utk',
                        help='Datasets would be used to train.')
    parser.add_argument('--pretrained_path', type=str,
                        help='Path to pretrained model.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='The input image size.')
    parser.add_argument('--ignore_list', type=str, default='ignore_list.txt',
                        help='The directory for saving checkpoints.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The training batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs would be trained.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='The initial learning rate.')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='The directory for saving checkpoints.')
    parser.add_argument('--save_history', action='store_true', default=True,
                        help='The flag indicates visualization.')
    return parser.parse_args(argv)


def mae(y_true, y_pred):
    classes = tf.range(101, dtype=tf.float32)
    y_true = tf.tensordot(y_true, classes, axes=1)
    y_pred = tf.tensordot(y_pred, classes, axes=1)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.function
def parse_fn(path, age, gender, image_size, augmentation=True):
    """Parse image function."""
    image = tf.io.decode_jpeg(tf.io.read_file(path, 'rb'), channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Data augmentation
    if augmentation:
        image = augmentor.random_flip_left_right(image)
        image = augmentor.random_rotate(image, angle=5, radian=False)
        image = augmentor.random_shift(image,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)
        image = augmentor.zoom_random(image, percentage_area=0.9)
        image = augmentor.random_contrast(image,
                                          min_factor=0.8,
                                          max_factor=1.2)
        image = augmentor.random_brightness(image,
                                            min_factor=0.8,
                                            max_factor=1.2)
        image = augmentor.random_erase(image, rectangle_area=0.15)

    image = image / 255.
    age = tf.one_hot(age, 101)
    return image, {'age': age, 'gender': gender}


def main(args):
    # Fixed parameters
    data_dir = args.data_dir
    ignore_list = args.ignore_list
    dataset = args.dataset
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    model_path = args.pretrained_path

    # Data preparation
    ((_, _, _), (val_paths, val_ages, val_genders)) = \
        utils.load_data(data_dir, dataset)

    appa_dir = os.path.join(data_dir, 'appa-real-release')
    appa_paths, appa_ages = utils.load_appa(appa_dir, ignore_list)

    train_paths = list(appa_paths)
    train_ages = list(appa_ages)
    num_train = len(train_paths)
    num_val = len(val_paths)

    train_genders = np.zeros((num_train, 1))

    print(f'Number of training examples: {len(train_paths)}')
    print(f'Number of validation examples: {len(val_paths)}')

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_ages, train_genders))
    train_data = train_data.shuffle(1000) \
        .map(lambda x, y, z: parse_fn(x, y, z, image_size),
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(tf.data.experimental.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_ages, val_genders))
    val_data = val_data.map(
        lambda x, y, z: parse_fn(x, y, z, image_size, False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # for batch in train_data.take(1):
    #     pass
    #
    # images = batch[0]
    # aug_images = batch[1]
    # ages = batch[2]['age']
    # for image, aug_image, age in zip(images[:10], aug_images[:10], ages[:10]):
    #     image = image.numpy() * 255.
    #     image = image.astype('uint8')
    #     aug_image = aug_image.numpy() * 255.
    #     aug_image = aug_image.astype('uint8')
    #     print(np.argmax(age.numpy()))
    #
    #     plt.subplot(121)
    #     plt.imshow(image)
    #     plt.subplot(122)
    #     plt.imshow(aug_image)
    #     plt.show()

    model = load_model(model_path)
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-2:]:
        if layer.name != 'gender':
            layer.trainable = True

    opt = Adam(learning_rate=lr)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(loss={'age': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'},
                  optimizer=opt,
                  metrics={'age': mae, 'gender': 'acc'})
    model.summary()

    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    model_name = 'age_only_model.{epoch:03d}.h5'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for saving the model and learning rate schedule
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_age_mae',
                                 verbose=1,
                                 save_best_only=True)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=3,
                                   verbose=1,
                                   min_lr=0.5e-6)
    early_stopping = EarlyStopping(monitor='val_age_mae',
                                   mode='auto',
                                   patience=5,
                                   verbose=1,
                                   restore_best_weights=True)
    callbacks = [checkpoint, lr_reducer, early_stopping]

    # Train the model
    hist = model.fit(
        train_data,
        steps_per_epoch=num_train // batch_size,
        validation_data=val_data,
        validation_steps=num_val // batch_size,
        epochs=epochs,
        callbacks=callbacks)

    if args.save_history:
        history_path = os.path.join(save_dir, 'pretrained_history.npy')
        np.save(history_path, hist.history)


if __name__ == '__main__':
    main(parse_args(argv[1:]))
