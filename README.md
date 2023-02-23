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
To build the image using Docker container execute the following commands into project root directory

Make sure Docker is installed
```
$ docker build -t alekhyadronavalli/furniture_classification:0.0.1.RELEASE .

Run command: docker-compose up

Note that if you make changes to Dockerfile or requirements.txt, you will need to run command: docker-compose build to rebuild the image.
After running the app, it should be located at http://localhost/ (port 8000). Note this is an override of the default port 5000 for Flask apps, but on M1 macs, port 5000 is sometimes used by a process.
To stop the running web server from terminal, press Control + C on the keyboard.
```
# Model 
To run this, download the model and place it in the project root directory with filename as `model.h5`
Download Link --> [my_model1.h5](https://drive.google.com/drive/folders/1MLgclJRSRJUKOftHP7wklIfZ95HThKj6?usp=sharing)

# Flask - Local Server
To start the local server using flask, use the following command
```
python app.py
```
# Outputs
![output](https://github.com/AlekhyaD/Identifying-Furnitures-using-Resnet/blob/main/Capture.PNG)
# To Train - Scratch
**Step 1:**
To train the model, download the following and place it under `data` folder,
Dataset - [Furniture Dataset]()
**Step 2:**
```
python train.py
```
