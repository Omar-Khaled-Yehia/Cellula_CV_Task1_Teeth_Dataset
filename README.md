# Cellula_CV_Task1_Teeth_Dataset
Teeth Dataset Classification with CNN
This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) to classify images of teeth into various categories. The project uses TensorFlow and Keras for building, training, and evaluating the model.

Table of Contents
Project Overview
Repository Contents
Data Preprocessing
Model Architecture
How to Use
Results
Acknowledgements
Project Overview
The goal of this project is to develop a deep learning model capable of classifying images of teeth into multiple predefined categories. The dataset used in this project is organized into training, validation, and testing sets.

Repository Contents
Cellula_Task1_Teeth_Dataset.ipynb: The main notebook that includes:
Data preprocessing and augmentation using ImageDataGenerator.
Definition of the CNN model architecture.
Visualization of sample images from the dataset.
Training the model with techniques to prevent overfitting.
Model evaluation on the test dataset.
Data Preprocessing
The dataset is structured into three directories: Training, Validation, and Testing.
Data augmentation techniques such as rotation, shifting, zooming, and brightness adjustments are applied to the training images to enhance model performance.
Model Architecture
The CNN model consists of several convolutional layers, followed by max-pooling layers.
Batch normalization and dropout layers are used for regularization and to improve training stability.
The model is compiled using categorical cross-entropy loss and the Adam optimizer.
How to Use
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/teeth-dataset-classification.git
cd teeth-dataset-classification
Open the Jupyter notebook Cellula_Task1_Teeth_Dataset.ipynb and execute the cells to preprocess data, train the model, and evaluate its performance.

Ensure that the following dependencies are installed:

bash
Copy code
pip install tensorflow matplotlib numpy
Results
The model is evaluated on the test dataset, and the performance metrics are visualized within the notebook.
Sample images from the training, validation, and testing datasets are displayed to provide insight into the model's predictions.
Acknowledgements
This project utilizes TensorFlow and Keras libraries for deep learning model development.

To use the app go to https://teeth-cellula.streamlit.app/
