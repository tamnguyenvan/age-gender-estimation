import os
import glob
import numpy as np
from datetime import datetime

import tensorflow as tf
from scipy.io import loadmat
from PIL import Image

np.random.seed(42)


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(data_dir, db='imdb', split=0.1):
    out_paths = []
    out_ages = []
    out_genders = []

    db_names = db.split(',')
    # Load utkface if need.
    if 'utk' in db_names:
        utk_dir = os.path.join(data_dir, 'utkface-new')
        utk_paths, utk_ages, utk_genders = load_utk(utk_dir)
        out_paths += utk_paths
        out_ages += utk_ages
        out_genders += utk_genders

    db_names.remove('utk')
    for d in db_names:
        image_dir = os.path.join(data_dir, '{}_crop'.format(d))
        mat_path = os.path.join(image_dir, '{}.mat'.format(d))
        full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, d)
        sample_num = len(face_score)
        min_score = 1.

        for i in range(sample_num):
            if face_score[i] < min_score:
                continue

            if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
                continue

            if ~(0 <= age[i] <= 100):
                continue

            if np.isnan(gender[i]):
                continue

            out_genders.append(int(gender[i]))
            out_ages.append(age[i])
            out_paths.append(os.path.join(image_dir, str(full_path[i][0])))

    indices = np.arange(len(out_paths))
    np.random.shuffle(indices)
    out_paths = list(np.asarray(out_paths)[indices])
    out_ages = list(np.asarray(out_ages)[indices])
    out_genders = list(np.asarray(out_genders)[indices])

    num_train = int(len(out_paths) * (1 - split))
    train_paths, train_ages, train_genders = out_paths[:num_train], out_ages[:num_train], out_genders[:num_train]
    val_paths, val_ages, val_genders = out_paths[num_train:], out_ages[num_train:], out_genders[num_train:]

    return (train_paths, train_ages, train_genders), (val_paths, val_ages, val_genders)


def load_utk(data_dir):
    """Load UTKFace dataset."""
    out_paths = []
    out_ages = []
    out_genders = []

    paths = glob.glob(os.path.join(data_dir, 'crop_part1', '*'))
    for path in paths:
        filename = os.path.basename(path)
        out_paths.append(path)
        age, gender = filename.split('_')[:2]
        age = int(age)
        gender = 1 if int(gender) == 0 else 0

        out_ages.append(age)
        out_genders.append(gender)
    return out_paths, out_ages, out_genders


def load_appa(data_dir, ignore_list_filename=None):
    """Load APPA-real dataset."""
    out_paths = []
    out_ages = []

    ignore_filenames = set()
    if ignore_list_filename is not None:
        ignore_list_path = os.path.join(data_dir, ignore_list_filename)
        ignore_filenames = set(x.strip() for x in open(ignore_list_path))

    data_file = os.path.join(data_dir, 'gt_avg_train.csv')
    image_dir = os.path.join(data_dir, 'train')
    with open(data_file) as f:
        lines = [x.strip() for x in f]
        for line in lines[1:]:
            filename, _, _, _, age = line.strip().split(',')
            if filename in ignore_filenames:
                continue
            image_path = os.path.join(image_dir, filename + '_face.jpg')
            age = int(age)

            out_paths.append(image_path)
            out_ages.append(age)
    return out_paths, out_ages


def load_aligned_data(data_dir, split=0.1):
    out_paths = []
    out_ages = []
    out_genders = []

    paths = glob.glob(os.path.join(data_dir, '*'))
    for path in paths:
        filename = os.path.basename(path)
        age, gender = filename.split('_')[-2:]
        gender = gender.split('.')[0]
        age = int(age)
        gender = int(gender)

        out_paths.append(path)
        out_ages.append(age)
        out_genders.append(gender)

    indices = np.arange(len(out_paths))
    np.random.shuffle(indices)
    out_paths = np.asarray(out_paths)[indices]
    out_ages = np.asarray(out_ages)[indices]
    out_genders = np.asarray(out_genders)[indices]

    num_train = int(len(out_paths) * (1 - split))
    train_paths, train_ages, train_genders = out_paths[:num_train], out_ages[:num_train], out_genders[:num_train]
    val_paths, val_ages, val_genders = out_paths[num_train:], out_ages[num_train:], out_genders[num_train:]

    return (train_paths, train_ages, train_genders), (val_paths, val_ages, val_genders)
