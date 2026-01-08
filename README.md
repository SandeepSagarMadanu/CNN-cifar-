# CNN Image Classification using CIFAR-10

This project is a hands-on implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.  
The main goal of this project is to understand how CNNs work in practice and how they learn visual patterns from image data.

---

## About the Project

In this project, I built a CNN model from scratch using TensorFlow and Keras.  
The model is trained to recognize everyday objects such as airplanes, cars, animals, and ships from small color images.

This project focuses on:
- Understanding CNN layers
- Learning image preprocessing techniques
- Training and evaluating a deep learning model
- Interpreting model predictions and errors

---

## Dataset Information

The project uses the CIFAR-10 dataset, which is commonly used for image classification tasks.

- Total images: 60,000  
- Image size: 32 × 32 (RGB)  
- Classes: 10  

### Class Labels
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

### Dataset Split
- Training images: 50,000  
- Testing images: 10,000  

The dataset is loaded directly using TensorFlow.

---

## Model Architecture

The CNN model is created using the Keras Sequential API.

It includes:
- Convolution layers to extract image features
- Max pooling layers to reduce image size
- Fully connected layers for classification
- A softmax output layer for predicting class probabilities

The architecture is intentionally kept simple to clearly understand the role of each layer.

---

## Data Preprocessing

Before training the model:
- Image pixel values are scaled from 0–255 to 0–1  
- Labels are reshaped for compatibility with the loss function  
- Sample images are visualized to verify data correctness  

These steps help improve training stability and performance.

---

## Model Training

The model is trained using:
- Optimizer: Adam  
- Loss function: Sparse Categorical Crossentropy  
- Metric: Accuracy  

Training is performed for multiple epochs so the model can gradually improve its predictions.

---

## Evaluation and Results

After training, the model is evaluated on unseen test data.

Evaluation includes:
- Test accuracy  
- Classification report (precision, recall, F1-score)  
- Confusion matrix to analyze class-wise predictions  

Some classes with similar visual features (such as cats and dogs) are harder to distinguish, which is expected for a basic CNN.

---

## Project Structure

CNN-CIFAR10/
│
├── CNN(cifar10).ipynb
├── README.md

---

## What I Learned

Through this project, I gained practical experience in:
- Building CNNs from scratch  
- Working with image datasets  
- Improving model performance through preprocessing  
- Understanding model predictions and errors  

This project helped strengthen my understanding of deep learning fundamentals.

---

## Future Improvements

Possible improvements include:
- Adding dropout layers to reduce overfitting  
- Using data augmentation  
- Applying batch normalization  
- Training with transfer learning models such as ResNet or MobileNet  

---

## Author

**Sandeep Sagar Madanu**  
AI & Data Science Enthusiast  

GitHub: https://github.com/SandeepSagarMadanu  
Portfolio: https://sandeepmadanu.netlify.app  

