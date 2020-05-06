# Welcome!

## About this talk:

There are lots of tutorials on making an object detector work with a pre-trained
dataset, but not many on how to make your own datasets for object detection.

Over coming Synthetic data

Wildlife detection and monitoring with a limited dataset that anyone with a camera or smart phone can create

I hope that no matter your experience you'll learn something new today!

## About me:
👋 Hello,  I'm [Sage Elliott](https://www.linkedin.com/in/sageelliott/).

I'm a technical evangelist at [Galvanize](https://www.galvanize.com/) with experience creating computer vision systems for manufacturing quality assurance, architecture design generation, and wildlife monitoring. I love helping people learn new things.

I'm really excited to have you here for this this talk!
Originally I was going to give this talk in person at a python meetup in
Seattle. Then 2020 happened... Hopefully doing this virtually will reach outside
of seattle! 

Where are you watching this from right now? 

**Thank you all for coming tonight!**



## Sponsorship & Give away

I was thinking of ways to bring in sponsors to virtual events. 

[hyperlabel](https://hyperlabel.com/) is the image labeling tool I used in this 
project and they agreed to be a sponsor!

Hyperlabel will be giving 4 winners $25 each for door dash to help support 
your favorite local restaurants. 

Enter to win here: [link]()

Thank you [hyperlabel](https://hyperlabel.com/)!!!

[![hyperlabel image labeling logo](pictures/hyperLabelLogo.png)](https://hyperlabel.com/)

--------------

# Data

The not always most fun, but maybe the most important. 

In this case I actually had a lot of fun!


## Collecting the data:

I used my smart phone.

Often you're going to reduce the resolution during your model training process, 
so super high resolution doesn't matter too much.

When collecting think of what you want to capture:

- Object angles
    - Side
    - top
    - back
    - front

- Object positions
    - sitting
    - swimming
    - eating
    - flying

- Object variations
    - age
    - color
    - type  

This example I am creating a dataset of canadian geese. Fortunately for me.
They don't have much variation in appearance.

but I still need to take in account the first two 

In total I only took 87 photos. Many were very similar.

> Example pictures:




> Not part of the data set, but the geese recently had babies!
![baby geese](pictures/baby2.jpg)
![baby geese](pictures/baby1.jpg)



## Overcome a limited dataset


### Synthetic data 

Creating the synthetic dataset

Different backgrounds, rotation, positions, object variations(cow example)

Example use cases of synthtic datasets


I think this idea is one of the coolest things, it's gaining traction
but I'm still surprised that it's not talked about more!

This example I made my synthetic dataset manually, but you're probably already
asking how can I automate it?

Hyper Label future feature?



we're going to come back to creating more extreme synthetic data
after our initial training. 


### data augmentation

You may already be familiar with a more widely used concept of data augmentation

What data augmentation is:

What it isn't:




## Resizing the dataset:

Even though you usually resize in during loading your dataset for training
it can help speed things up resizing your images before loading into memory. 

Resize script:

```
Resize script here
```


## Labeling the data 

Different type of labels. 

- Image segementation
- Object detection
- Classification

Some other label options you may see in computer vision
- Keypoint
- context



Our case we want to do object detection. The boxes around the objects.

There are a couple good labeling options

- hyperlabel
- imagelabeler

I chose hyper label. Again shout out for them sponsoring tonight!

Enter give away here [link]()

keep in mind that every labeler may have slightly different annotation generation


What labeling looks like

Understanding the annotations

Export:
You get different export options

For me I'm using VOC pascal which exports the images and XML annotations of
boudning boxes

XML Example:

```xml
<annotation>
  <folder>GeneratedData_Train</folder>
  <filename>3.png</filename>
  <source>
    <database>3</database>
  </source>
  <size>
    <width>800</width>
    <height>600</height>
    <depth>Unknown</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>goose</name>
    <pose>Unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occluded>0</occluded>
    <bndbox>
      <xmin>159.28430879566355</xmin>
      <xmax>342.9821219169359</xmax>
      <ymin>219.6319686872721</ymin>
      <ymax>405.6469286512451</ymax>
    </bndbox>
  </object>
  <object>
    <name>goose</name>
    <pose>Unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occluded>0</occluded>
    <bndbox>
      <xmin>537.1769383697813</xmin>
      <xmax>660.7554380746769</xmax>
      <ymin>55.55722749247779</ymin>
      <ymax>179.5672008017932</ymax>
    </bndbox>
  </object>
  <object>
    <name>goose</name>
    <pose>Unknown</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occluded>0</occluded>
    <bndbox>
      <xmin>84.85088007113569</xmin>
      <xmax>139.24456192532307</xmax>
      <ymin>4.522350222409504</ymin>
      <ymax>82.74403297229314</ymax>
    </bndbox>
  </object>
</annotation>
```

Converting to a CSV file.

You could skip the step of generating a CSV file and directly create a TF Record
but I've found having a CSV file helpful in the past.

- A chance to pause and check your data
- If your labeling tool doesn't save a project, you can append new annotation
to your CSV file.


```python
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(round(float(member[5][0].text))),
                     int(round(float(member[5][1].text))),
                     int(round(float(member[5][2].text))),
                     int(round(float(member[5][3].text)))
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'xmax', 'ymin', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')
main()

```


#### OpenCV annotation check

Always check your annotations! I really wish someone had drilled this into me
early. checking you read in your annotations correctly can save you a lot of
time debugging.

Use openCV to read the annotations

```python
# %%
import cv2
import pandas as pd
from PIL import Image

# %%
full_labels = pd.read_csv('train_labels.csv')

# %%
full_labels.head(10)

# %%
def draw_boxes(image_name):
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread('train/{}'.format(image_name))
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 3)
    return img

# %%
Image.fromarray(draw_boxes('20200320_180628.jpg'))

# %%
Image.fromarray(draw_boxes('20200320_180651.jpg'))
```

![reading in XML wrong](pictures/wrong_read.png)

![reading in XML right](pictures/right_read.png)


I'm guilty of not checking and wasting hours debugging. because I was "sure"
I was reading them correctly.


# Training 

## Model Selection

Different types of computer vision applications require different models

#### Model types

SSD
Yolo
Mask R-CNN
CNN

#### Transfer learning

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

I chose to use ones included in tensorflows offcial repo. I feel like it's
maybe the most approachable for most people

Note that most of the models are under the research directory. These are not
always offcially maintained. 

There is a great resource to get started with the included tensflow model here:

Setup tips Quick summary of setting up:

TF & Python versions


More resources included at the end.

## Training the model

changes to work with different annotations

I started with default settings

But you can change them here:


# Results



Example without synthetic dataset:

Let's say now our goal is to detect geese in my apartment

I'm a goose!

transfer learning we didn't do a good enough job of telling it what WASN't a
goose




Example with synthetic dataset:


## Retrainig with synthtic dataset of living room

Even with out small dataset we did a pretty good job of telling our model
what a goose IS.

But we didn't do a good job of telling it what a goose ISN"T.

We want to add in some noise to the data, like objects and people so as the
model is training it can learn when it makes a mistake on them.



## Future improvements:

Make much more synthetic data with python


# Wrap up

## Summary

I hope this inspired you to make your own object detector or get started with computer vision!

And how powerful sythetic data sets have the potential to be!

## Stay connected:

Please feel free to reach out to me with any questions. 
I love helping other learn.

linkedin
twitter
email
site

## Hyper Label Give away

Again, thank you to hyper label fo sponsoring

## Useful Resources:

- [Hyper Label: Image labeling]() used for labeling the images

- [Tensorflow object detection setup guide](https://gilberttanner.com/blog/creating-your-own-objectdetector)

- [Another good tensorflow object tutorial](https://www.youtube.com/watch?v=Rgpfk6eYxJA&t=1024s)

- [Make synthetic datsets with python](https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets)

- [Racoon detection blog post](https://github.com/datitran/raccoon_dataset)

Scripts I used: