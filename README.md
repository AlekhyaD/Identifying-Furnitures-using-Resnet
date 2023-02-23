# Identifying-Furnitures-using-Resnet

# About the project
The objective of this project is to use deep learning techniques to accurately identify different types of furniture items such as chairs, beds, and sofas from a given image. In order to achieve this goal, we have leveraged the power of Convolutional Neural Network (CNN) architecture, which is a deep learning technique specifically designed for image recognition tasks.

Our approach involves using the ResNet-152 as backbone model to analyze and process the input image through a series of convolutional and pooling layers. This process helps to extract relevant features from the image and learn high-level representations that can be used for classification. After convolving the image, the resulting feature maps are flattened and passed through a softmax function to produce a prediction on the class of the furniture item. We achieved an accuracy 90% on the validation dataset in detecting furniture items from images. 

# Environment Setup
Clone this repo:
```
git clone https://github.com/andreiliphd/deploying-neural-network-flask.git
```
Install all the dependencies using docker.
### Docker
To run this code using Docker container execute the following commands into project root directory
```
$ docker build -t python-neural-network .
$ docker run -p 8080:8080 -d python-neural-network
```
# Model 
To run this, download the model and place it in the project root directory with filename as `model.h5`
Download Link --> [model.h5]()

# Flask - Local Server
To start the local server using flask, use the following command
```
python app.py
```
# Outputs

# To Train - Scratch
**Step 1:**
To train the model, download the following and place it under `data` folder,
Dataset - [Furniture Dataset]()
**Step 2:**
```
python train.py
```
