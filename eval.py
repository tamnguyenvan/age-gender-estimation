import os
import glob
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from keras.models import load_model
from keras.utils.data_utils import get_file
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
K.set_session(tf.Session(config=config))


def main():
    # load model
    MODEL_PATH = 'saved_models/age_gender_model.10-3.80.hdf5'
    img_size = 64
    batch_size = 32
    data_dir = 'data'

    model = load_model(MODEL_PATH)
    dataset_root = os.path.join(data_dir, 'appa-real-release')
    validation_image_dir = os.path.join(dataset_root, 'valid')
    gt_valid_path = os.path.join(dataset_root, 'gt_avg_valid.csv')
    image_paths = glob.glob(os.path.join(validation_image_dir, '*_face.jpg'))

    faces = np.empty((batch_size, img_size, img_size, 3))
    ages = []
    image_names = []

    for i, image_path in tqdm(enumerate(image_paths)):
        image = np.asarray(Image.open(image_path).resize((img_size, img_size)), dtype='float32')
        faces[i % batch_size] = image
        image_names.append(os.path.basename(image_path)[:-9])

        if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
            faces = faces.astype('float32') / 255.
            results = model.predict(faces)
            ages_out = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[0].dot(ages_out).flatten()
            ages += list(predicted_ages)

    name2age = {image_names[i]: ages[i] for i in range(len(image_names))}
    df = pd.read_csv(str(gt_valid_path))
    appa_abs_error = 0.0
    real_abs_error = 0.0

    for i, row in df.iterrows():
        appa_abs_error += abs(name2age[row.file_name] - row.apparent_age_avg)
        real_abs_error += abs(name2age[row.file_name] - row.real_age)

    print("MAE Apparent: {}".format(appa_abs_error / len(image_names)))
    print("MAE Real: {}".format(real_abs_error / len(image_names)))


if __name__ == '__main__':
    main()
