## My goals:

My Test Case:

Geese

Lots of tutorials on making an object detector

not many on making your own dataset and how to work with it for different programs


## Collecting the data:

I used my smart phone

Often you reduce the resolution during your model training process, so super high resolution doesn't matter too much


## Model Selection

Different types of computer vision applications require different models

#### Model types

SSD
Yolo
Mask - R-CNN
CNN

#### Transer learning

CNNs basics
Weights
Imagenet dataset
Coco dataset

Minimum 200 images rule of thumb for transfer learning
May need thousands without transfer learnin

#### Implementations

I chose to use ones included in tensorflows repo

TF & Python versions


## Overcome a limited dataset

Creating the synthetic dataset

Different backgrounds, rotation, positions, object variations(cow example)

## Labeling the data 

Different type of labels. 

- Image segementation
- Object detection
- Classification


I chose hyper label

Always check your annotations!

#### OpenCV annotation check

Show script 

Use openCV to read the annotations

Not doing this has cost me hours before... Don't do the same thing....

## Training the model

I used models in tensorflows library

Tuning & What I learned during model training

tutorial here

changes to work with different annotations


## Running the model / Deploying

Results:

example without my living room


## Future improvements:

Make synthtic datsets with python

Deploy on pi

TF JS

## Stay connected:

linkedin
twitter
email
site

## Resources:

I hope this inspired you to make your own object detector!

Scripts I used:

Useful Instructions:
https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets

Make synthtic datsets with python

