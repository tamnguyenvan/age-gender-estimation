# Age and Gender estimation
Age and gender estimation using CNN. We train the model on [the IMDB and Wiki dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). We tried variety of deep network architectures like ResNet, DenseNet, Inception-ResNet-v3 to find out which one is the best. Eventually, we decide to pick ResNet as our model.

# Dependencies
- tensorflow >=1.15
- dlib (for face detection)
- opencv-python (for camera demo)

# Train the model
At first, download face-only datasets, extract and put them anywhere you'd like. Afterthat, run training script `python train.py`. Show help for more options.

# Test the model
We also provided a pretrained model to help you instantly test it. If you'd like to test on a single image, run `python test_on_image.py --model_path MODEL_PATH --image_path IMAGE_PATH`. Otherwise, in case you'd like to test on realtime camera, run `python test_on_camera.py --model_path MODEL_PATH --image_path IMAGE_PATH`. Enjoy!

# Contact
Any question could be left as issues. Contact me via email tamvannguyen200795@gmail.com. You're all welcome.
