"""Evaluate our model on APPA-real dataset."""
import os
import glob
import argparse
from sys import argv

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# Fix tensorflow bug on rtx card
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)


def parse_args(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='The input image size.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to appa-real directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Evaluation batch size.')
    return parser.parse_args(argv)


def main(args):
    if args.model_path is None and not os.path.exists(args.model_path):
        print('Not found model path.')
        return

    image_size = args.image_size
    data_dir = args.data_dir
    batch_size = args.batch_size

    # Load image paths
    if data_dir is None:
        print('Not found data dir.')
        return

    image_dir = os.path.join(data_dir, 'appa-real-release')
    if not os.path.exists(image_dir):
        print('Not found image dir.')
        return

    valid_image_dir = os.path.join(image_dir, 'valid')
    gt_valid_path = os.path.join(image_dir, 'gt_avg_valid.csv')
    image_paths = glob.glob(os.path.join(valid_image_dir, '*_face.jpg'))

    # Load the model and run prediction
    model = load_model(args.model_path)
    faces = np.zeros((batch_size, image_size, image_size, 3))
    ages = []
    image_names = []
    from matplotlib import pyplot as plt
    for i, image_path in enumerate(image_paths):
        image_names.append(os.path.basename(image_path)[:-9])

        image = Image.open(image_path).resize(size=(image_size, image_size))
        plt.imshow(np.array(image))
        plt.show()
        break
        image_np = np.array(image, dtype='float32')
        image_np /= 255.
        faces[i%batch_size] = image_np
        if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
            y_pred = model.predict(faces)
            classes = np.arange(0, 101).reshape(-1, 1)
            age_pred = y_pred[0].dot(classes).flatten()
            ages += list(age_pred)

    # Compute mean absolute error
    name2age = {image_name: age for image_name, age in zip(image_names, ages)}
    appa_abs_error = 0.
    real_abs_error = 0.

    df = pd.read_csv(gt_valid_path)
    for i, row in df.iterrows():
        appa_abs_error += abs(name2age[row.file_name] - row.apparent_age_avg)
        real_abs_error += abs(name2age[row.file_name] - row.real_age)

    print('MAE apparent: {}'.format(appa_abs_error / len(image_names)))
    print('MAE real: {}'.format(real_abs_error / len(image_names)))


if __name__ == '__main__':
    main(parse_args(argv[1:]))
