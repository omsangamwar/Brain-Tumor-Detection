# Brain Tumor detection

## Introduction
Brain tumor detection is a critical task in medical imaging with profound implications for patient diagnosis and treatment planning. Traditional methods of tumor detection rely heavily on manual inspection by radiologists, which can be time-consuming and prone to human error. With the advancements in deep learning techniques and the availability of large medical imaging datasets, there has been a surge in the development of automated brain tumor detection systems.

## Objective
The primary objective of this project is to develop a deep learning model capable of accurately detecting brain tumors from medical imaging data such as Magnetic Resonance Imaging (MRI) scans. By leveraging the power of deep learning algorithms, we aim to create a system that can assist healthcare professionals in the early and accurate diagnosis of brain tumors, leading to better patient outcomes.

## Dataset
![Brain_Tumor_dataset](https://github.com/omsangamwar/Brain-Tumor-Detection/assets/117922569/3b2e282e-ba91-44b9-8519-e141f94f9064)


## Data Preprocessing

The dataset consists of MRI (Magnetic Resonance Imaging) scans of the brain, categorized into two classes: images containing brain tumors and images without tumors (healthy brain tissue). Each image is labeled accordingly, providing supervision for our deep learning model.

## a. Image Resizing:
MRI scans may come in various resolutions. To standardize the input size for our deep learning model, we resize all images to a consistent resolution (e.g., 256x256 pixels)

## b.Sharpening
Image sharpening is a technique used in digital image processing to enhance the clarity and detail of an image. It works by increasing the contrast along edges within the image, which makes them appear sharper to the human eye.

![Sharpening](https://github.com/omsangamwar/Brain-Tumor-Detection/assets/117922569/f0db9273-4aba-446f-bb2c-4edb438cbc92)

## c.Data Augmentation

#### Rotation Range: 
This parameter specifies the range of angles by which the original image can be rotated. 
#### Shear Range: 
Shear transformation distorts the shape of the image along one axis. 
#### Zoom Range: 
Zooming alters the scale of the image, either zooming in to focus on a specific area or zooming out to capture more of the surroundings. 
#### Horizontal and Vertical Flips: 
These parameters control whether horizontal and/or vertical flips are applied to the images. 
#### Width and Height Shift Range: 
These parameters specify the range by which the images can be horizontally and vertically shifted, respectively. 
#### Fill Mode: 
Fill mode determines how pixels outside the boundaries of the transformed image are filled. Options include constant, nearest, reflect, and wrap. 

![Data Augmented](https://github.com/omsangamwar/Brain-Tumor-Detection/assets/117922569/ee9c81e1-e15b-402e-91a3-63f8e292f43b)





