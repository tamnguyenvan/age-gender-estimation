import os
import glob
import numpy as np
import keras
from datetime import datetime
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
    valid_sample_num = 0
    for d in db.split(','):
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
            valid_sample_num += 1
        
    indices = np.arange(len(out_paths))
    np.random.shuffle(indices)
    out_paths = np.asarray(out_paths)[indices]
    out_ages = np.asarray(out_ages)[indices]
    out_genders = np.asarray(out_genders)[indices]

    num_train = int(len(out_paths) * (1 - split))
    train_paths, train_ages, train_genders = out_paths[:num_train], out_ages[:num_train], out_genders[:num_train]
    val_paths, val_ages, val_genders = out_paths[num_train:], out_ages[num_train:], out_genders[num_train:]

    print('Train/Val', len(train_paths), len(val_paths))
    return (train_paths, train_ages, train_genders), (val_paths, val_ages, val_genders)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths, ages, genders, target_size=(224, 224), batch_size=32, shuffle=True, dtype='float32'):
        self.data = list(zip(paths, ages, genders))
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.on_epoch_end()
    
    def __len__(self):
        return (np.floor(len(self.data) / float(self.batch_size))).astype(np.int)
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.data)
  
    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size: (idx+1) * self.batch_size]
        image_batch = np.empty((self.batch_size, *self.target_size, 3), dtype=self.dtype)
        age_batch = np.empty((self.batch_size, 1), dtype='uint8')
        gender_batch = np.empty((self.batch_size, 1), dtype='uint8')
        for i, (path, age, gender) in enumerate(batch):
            image = np.asarray(Image.open(path).resize(self.target_size), dtype=self.dtype)
            if len(image.shape) != 3:
                image = np.expand_dims(image, -1)
            image_batch[i] = image
            age_batch[i, 0] = age
            gender_batch[i, 0] = gender
        
        age_batch = keras.utils.to_categorical(age_batch, 101)
        # gender_batch = keras.utils.to_categorical(gender_batch, 1)
        image_batch /= 255.
        return image_batch, {'age': age_batch, 'gender': gender_batch}