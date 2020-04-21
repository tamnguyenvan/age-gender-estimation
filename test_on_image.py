"""
"""
import argparse
from sys import argv
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
tf.Session(config=config)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help='Path to the model.')
    parser.add_argument('--image_path', type=str,
                        help='Path to the image should be predicted.')
    parser.add_argument('--margin', type=float, default=0.4,
                        help='Face margin percentage.')
    return parser.parse_args(argv)


def imcrop(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2



def main(args):
    model = load_model(args.model_path)
    detector = dlib.get_frontal_face_detector()
    input_dim = (112, 112)

    margin = args.margin
    image_path = args.image_path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image, 1)
    faces = []
    bboxes = []
    for d in dets:
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = int(x1 - margin * w)
        yw1 = int(y1 - margin * h)
        xw2 = int(x2 + margin * w)
        yw2 = int(y2 + margin * h)
        cropped_img = imcrop(image, xw1, yw1, xw2, yw2)
        face = cv2.resize(cropped_img, (input_dim))
        faces.append(face)
        bboxes.append((x1, y1, x2, y2))

    if faces:
        faces = np.asarray(faces, dtype='float32') / 255.
        pred = model.predict(faces)
        ages_out = np.arange(0, 101).reshape(101, 1)
        pred_ages = list(pred[0].dot(ages_out).flatten().astype('uint8'))
        pred_genders = np.argmax(pred[1], axis=1)

        for bbox, age, gender in zip(bboxes, pred_ages, pred_genders):
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            text = 'Age: {} Gender: {}'.format(age, 'male' if gender else 'female')
            cv2.putText(image, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    out_path = image_path.split('.')[0] + '_out.jpg'
    cv2.imwrite(out_path, image)


if __name__ == '__main__':
    main(parse_args(argv[1:]))