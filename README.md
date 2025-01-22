

```markdown
# Convolutional Neural Network (CNN) for Image Classification

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The project includes data preprocessing, model building, training, and prediction.

## Requirements

Ensure you have Python 3.x installed. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Project Structure

The project assumes the following directory structure for the dataset:

```
dataset/
│-- training_set/
│   │-- cats/
│   │-- dogs/
│-- test_set/
│   │-- cats/
│   │-- dogs/
│-- single_prediction/
    │-- cat_or_dog_1.jpg
    |-- cat_or_dog_2.jpg
```

## How to Run

1. Set up the dataset following the structure above.
2. Open the Python script or Jupyter Notebook.
3. Run the code to train the CNN model.
4. Test the model by predicting an image from the `single_prediction` directory.

## Features

- **Data Preprocessing:** Image augmentation using `ImageDataGenerator`.
- **Model Building:** A sequential CNN with convolutional layers, max pooling, flattening, and dense layers.
- **Training:** Binary classification using the Adam optimizer and binary cross-entropy loss function.
- **Prediction:** Classifying a new image as either a cat or a dog.

## Dependencies

- TensorFlow
- Keras
- NumPy

## Model Training

The model is trained with:

- **Epochs:** 25
- **Batch Size:** 32
- **Input Image Size:** 64x64 pixels
- **Activation Functions:** ReLU and Sigmoid
