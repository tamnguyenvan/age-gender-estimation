import os
import argparse
from sys import argv

import numpy as np
from matplotlib import pyplot as plt


def parse_args(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='Path to saving checkpoints.')
    parser.add_argument('--save_fig', action='store_true',
                        help='The flag indicates visualization.')
    parser.add_argument('--output_path', type=str, default='result.png',
                        help='The learning curve figure output path.')
    return parser.parse_args(argv)


def plot_result(history, output_path=None):
    """Plot the training result."""
    history = history.tolist()

    epochs = len(history['age_mae'])
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.plot(history['age_mae'], label='age mae')
    plt.plot(history['val_age_mae'], label='val age mae')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.xticks(np.arange(0, epochs, 5))
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(222)
    plt.plot(history['age_loss'], label='age loss')
    plt.plot(history['loss'], label='loss')
    plt.plot(history['gender_loss'], label='gender loss')
    plt.plot(history['val_age_loss'], label='val age loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.plot(history['val_gender_loss'], label='val gender loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, 5))
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(223)
    plt.plot(history['gender_acc'], label='gender accuracy')
    plt.plot(history['val_gender_acc'], label='val gender accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(0, epochs, 5))
    plt.legend(loc='best')
    plt.grid()

    plt.suptitle('Training Result')
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

if __name__ == '__main__':
    args = parse_args(argv[1:])

    output_path = args.output_path
    input_path = args.input_path
    save_fig = args.save_fig

    hist = np.load(input_path, allow_pickle=True)
    if save_fig:
        plot_result(hist, output_path)
    else:
        plot_result(hist)
