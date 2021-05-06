# AI-GAN-2021

## Requirements are detailed in requirements.txt
You may not need all of them, but it's the environment I used.

## DCGAN
To Train
Run "Own GAN/train.py"
This'll create 100 epochs of stuff by default (Which should take less than 1 hour)

To Test
Run "Own GAN/test.py" to generate an image
There will also be results in "Own GAN/Output" for each epoch, as well as checkpoints at each 5th epoch

If you wish to configure this Stage, the settings are all at the top of "Own GAN/main.py"

## CycleGAN
To Train Sea
Run "CycleGAN/train.py" with the arguments "--dataroot ./TrainingImages/Sea --name sea_cyclegan --model cycle_gan --preprocess crop --crop_size 256 --display_id 0"
To Train GAN
Run "CycleGAN/train.py" with the arguments "--dataroot ./TrainingImages/GAN --name gan_cyclegan --model cycle_gan --preprocess crop --crop_size 128 --display_id 0"

To Test Sea
Run "CycleGAN/test.py" with the arguments "--dataroot ./TestImages --name sea_cyclegan --model cycle_gan --no_dropout"
To Test GAN
Run "CycleGAN/test.py" with the arguments "--dataroot ./TestImages --name gan_cyclegan --model cycle_gan --no_dropout"

In this current implementation, you need to 
Place the DCGAN output into "CycleGAN/TestImages/testA" and then run both Test Models

### For convenience there are some example images already present, and the CycleGAN Models are both Trained

