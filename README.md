# Handwritten Digit Recognition using TensorFlow/Keras

## Overview

This project implements a neural network model to recognize handwritten digits using TensorFlow and Keras. It trains the model on the MNIST dataset, which is widely used for benchmarking image classification algorithms.

## Features

- Loads and preprocesses the MNIST dataset.
- Builds a neural network model with layers for flattening, dense layers with ReLU activation, and a softmax layer for digit classification.
- Trains the model using Adam optimizer and evaluates its accuracy on test data.
- Includes functionality to predict digits from user-provided images using the trained model.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   .\venv\Scripts\activate    # Windows
   source venv/bin/activate   # macOS/Linux
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
4. **Run the main script to train the model and predict digits:**
   ```sh
   python main.py

## Usage
- main.py: Runs the training process on the MNIST dataset and provides an interface to predict digits from images.
- handwritten_digits/: Directory to store user-provided images of handwritten digits for prediction.

## Credits
- This project uses the MNIST dataset available in TensorFlow/Keras for training and evaluation.
