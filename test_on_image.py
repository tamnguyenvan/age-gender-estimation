"""
"""
import argparse
from sys import argv
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
tf.compat.v1.Session(config=config)


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


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def main(args):
    model = load_model(args.model_path)
    detector = MTCNN()
    input_dim = (224, 224)

    margin = args.margin
    image_path = args.image_path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dets = detector.detect_faces(image)
    faces = []
    bboxes = []
    classes = np.arange(0, 101).reshape(101, 1)
    for d in dets:
        box = d['box']
        # x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        x1, y1, x2, y2, w, h = box[0], box[1], box[0] + box[2], box[1] + box[3], box[2], box[3]
        xw1 = int(x1 - margin * w)
        yw1 = int(y1 - margin * h)
        xw2 = int(x2 + margin * w)
        yw2 = int(y2 + margin * h)
        new_w, new_h = yw2 - yw1, xw2 - xw1
        new_size = max(new_w, new_h)
        pad_w = new_size - new_w
        pad_h = new_size - new_h
        cropped_img = imcrop(image, xw1 - pad_w // 2,
                             yw1 - pad_h // 2,
                             xw1 + new_size, yw1 + new_size)
        face = cv2.resize(cropped_img, input_dim)
        faces.append(face)
        bboxes.append((x1, y1, x2, y2))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if faces:
        faces = np.asarray(faces, dtype='float32') / 255.
        pred = model.predict(faces)
        pred_ages = list(pred[0].dot(classes).flatten().astype('uint8'))
        pred_genders = np.where(pred[1] > 0.5, 1, 0)

        for bbox, age, gender in zip(bboxes, pred_ages, pred_genders):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 153, 0), 2)
            text = '{}, {}'.format(age, 'M' if gender else 'F')
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 153, 0), 1, cv2.LINE_AA)

    cv2.imshow('test', image)
    cv2.waitKey(0)
    out_path = image_path.split('.')[0] + '_out.jpg'
    cv2.imwrite(out_path, image)


if __name__ == '__main__':
    main(parse_args(argv[1:]))
