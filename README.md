# Face Mask Detection - Real-time Detection
## Getting Started
In these tough COVID-19 times, wouldn’t it be satisfying to do something related to it? I decided to build a simple and basic Convolutional Neural Network (CNN) model using Pytorch and OpenCV to detect if you are wearing a face mask to protect yourself. Interesting! Let's start!
## Requirements
- Python 3.7
- OpenCV 3.4.2
- Torch 1.4.0 (GPU)
- Torchvision 0.5.0
- Numpy, Matplotlib
## Preprocess
Dataset I searched online, there were about 3,600 images including 1800 have masks and 1800 without masks, but most of them were not cut face, the images including the body. So we need to preprocess the data: crop face from images and eliminate noise data.
- Use haar-like feature to separate faces from images, then overwrite the old images
- For blurry, unclear images, delete it in order to the model is not mistaken or wrong
- After preprocessing the data, there were only 2760 images left
```
python Data_processing.py
```
## Train model
Divide the dataset into training  set and validation set (80/20)
- *Using CrossEntropyLoss, SGD(Stochastic Gradient Descent) lr = 0.01*
```
python train.py
```
## Result
Because  having only 2 classes, the dataset is preprocessed so the accuracy is high on training set and validation set (training 32 epochs).
#### *Accuracy chart*
<p align="center"> <img src="https://github.com/manhminno/Face-Mask-Detection/blob/master/Acc_chart.png"></p>

## Enjoy result:
```
* For webcam: python main.py --mode Webcam
* For image: python main.py --mode Image --path (Img_file)
```
## Demo
#### *Image*
<p align="center"> <img src="https://github.com/manhminno/Face-Mask-Detection/blob/master/Demo_img.jpg"></p>

#### *Video*
