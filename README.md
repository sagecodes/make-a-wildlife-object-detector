## My goals:

My Test Case:

Geese

Lots of tutorials on making an object detector

not many on making your own dataset and how to work with it for different programs


## Collecting the data:

I used my smart phone

Often you reduce the resolution during your model training process, so super high resolution doesn't matter too much

Think of angles

Think of positions 

Think of variations

This example I am creating a dataset of canadian geese. Fortunately for me.
They don't have much variation in appearance.

but I still need to take in account the first two 

pics:


During this time I 

Baby geese picture!

Resizing the dataset:

Even though you usually resize in during loading your dataset for training
it can help speed things up resizing your images before loading into memory. 

Resize script:


## Overcome a limited dataset

### Synthetic data 

Creating the synthetic dataset

Different backgrounds, rotation, positions, object variations(cow example)

This example I made my synthetic dataset manually, but you're probably already
asking how can I automate it?

Hyper Label future feature?



### data augmentation

You may already be familiar with a more widely used concept of data augmentation

What data augmentation is:

What it isn't:


## Labeling the data 

Different type of labels. 

- Image segementation
- Object detection
- Classification

Some other label options you may see in computer vision
- Keypoint
- context

Our case we want to do object detection. The boxes around the objects.

There are several labeling options

- hyperlabel
- imagelabeler

I chose hyper label

keep in mind that every labeler may have slightly different annotation generation

Labeling types:

Bounding box
Image segmentation
classification


What labeling looks like




Understanding the annotations

#### OpenCV annotation check

Always check your annotations! I really wish someone had drilled this into me
early. checking you read in your annotations correctly can save you a lot of
time debugging.

Use openCV to read the annotations

```
Show script 
```

I'm guilty of not checking and wasting hours debugging. because I was "sure"
I was reading them correctly.

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

May need thousands without transfer learning. Keep in mind that synthetic data 
may be a way to turn hundreds into thousands. Depends on you data and what
you're doing with it. 

#### Implementations

There are quite a few implementations for different model types. 

I chose to use ones included in tensorflows offcial repo

Note that most of the models are under the research directory. These are not
always offcially maintained. 

TF & Python versions

There is a great resource to get started with the included tensflow model here:

Quick summary of setting up:


More resources included at the end.

## Training the model

changes to work with different annotations

I started with default settings

But you can change them here:



Example without synthetic dataset:

I'm a goose!



Example with synthetic dataset:




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
Step by step tutorials for setting up a tensorflow Object detector:
https://gilberttanner.com/blog/creating-your-own-objectdetector
https://www.youtube.com/watch?v=HjiBbChYRDw

Another good tensorflow object tutorial
https://www.youtube.com/watch?v=Rgpfk6eYxJA&t=1024s

Make synthtic datsets with python
https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets

Racoon datset exmaple:
https://github.com/datitran/raccoon_dataset
