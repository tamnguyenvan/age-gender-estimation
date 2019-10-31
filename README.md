# Age Gender Estimation
Keras implementation for age and gender estimation. We use [the IMDB and Wiki dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
for training on.

# Dependencies
- `Keras==3.1.1`
- `tensorflow-gpu==1.14.0`
- `dlib` (for demo)
- `opencv`

# Usage
## Download
First, download dataset and put to your own directory.

## Training
Next, edit `train.py` file to fit your demand then run training: `python train.py`

## Test
You can run `python eval.py` for testing phase. We test the model on `appa-real` dataset.

## Run on camera
We provide the script to help you run the model on realtime camera (opencv required).