# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
To Develop a convolutional deep neural network (CNN) to classify hand-written digits. The goal is to accurately identify digits from 0 to 9 based on scanned images of handwritten digits. Additionally, the model should be capable of handling scanned handwritten images that are not part of the standard dataset.

The MNIST dataset stands as a cornerstone in both machine learning and computer vision, offering a standardized benchmark for evaluating models. Comprising 28x28 pixel grayscale images depicting handwritten digits from 0 to 9, it includes a meticulously divided training set of 60,000 images and a test set of 10,000 images. Each image's grayscale representation spans pixel values from 0 to 255, where 0 signifies black and 255 denotes white. Researchers and practitioners extensively utilize this dataset to train and assess a multitude of machine learning models, particularly focusing on digit recognition tasks. Leveraging MNIST, we aim to develop and scrutinize a convolutional deep neural network tailored specifically for digit classification while also assessing its adaptability and generalization capabilities through real-world scanned handwritten images not present in the dataset.

## Neural Network Model
![0](https://github.com/user-attachments/assets/f57dc67b-5e9e-4785-b76d-44cc8b972c48)



## DESIGN STEPS

### STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

### STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

### STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.

### STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

### STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.


## PROGRAM

### Name:Thirugnanamoorthi G
### Register Number:212221230117
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image = X_train[0]
single_image.shape
plt.imshow(single_image, cmap='gray')
y_train.min()
X_train.max()
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
X_test_scaled = X_test / 255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train, 10)
y_test_onehot = utils.to_categorical(y_test, 10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image, cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1, 28, 28, 1)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28, 1)

model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 

model.fit(X_train_scaled, y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled, y_test_onehot))


metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/seven.png.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy() / 255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
    axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28), cmap='gray')

img_28_gray_inverted = 255.0 - img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy() / 255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
    axis=1)
print(x_single_prediction)


```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![1](https://github.com/user-attachments/assets/ac15e760-d0ff-4985-acdd-c7b1e893af70)

![2](https://github.com/user-attachments/assets/bd102202-f79f-4fc9-9188-75f8f5a50dfe)




### Classification Report

![5](https://github.com/user-attachments/assets/1c0f5f65-bcfa-43ce-b323-3957ffc0719b)



### Confusion Matrix

![4](https://github.com/user-attachments/assets/a10b199b-5987-465a-8e02-02d425b2f544)


### New Sample Data Prediction

![6](https://github.com/user-attachments/assets/018f5efd-dfb6-4bff-bed2-4bc2cfb4d616)

![7](https://github.com/user-attachments/assets/e1d23f82-ecd8-4ef8-83cd-3955a8693d0d)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
