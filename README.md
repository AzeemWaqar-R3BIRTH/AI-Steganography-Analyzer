
# Steganography Detection Tool - Artificial Intelligence Semester Project

A GUI-based application for detecting steganography in images. This tool allows users to train a machine learning model on clean and steganographic images, make predictions on new images, and visualize model performance. The application uses color histogram features and a Random Forest classifier to identify hidden content in images.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Description

The **Steganography Detection Tool** is a machine learning-based application designed to detect steganography in digital images. It utilizes a Random Forest classifier trained on image color histograms to identify hidden content in images. The application features an intuitive GUI that allows users to:

- Train the model on clean and steganographic images.
- Predict if new images contain hidden data.
- Visualize performance metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

The tool is built using Python, Tkinter for the GUI, OpenCV for image processing, and Scikit-learn for machine learning.

## Installation

Follow these steps to install and run the Steganography Detection Tool:

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/AI-Steganography-Analyzer.git
cd AI-Steganography-Analyzer
```

### 2. Install the required packages:
Make sure you have Python 3.6 or higher installed, then install the necessary packages with:
```bash
pip install -r requirements.txt
```
This will install all the dependencies for the project.

### 1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Run the application
```bash
python main.py
```

## Using the GUI

### Train Model:
- Click on the "Train Model" button.
- Select two directories: one for clean images and one for steganographic images. The application will use these images to train the model.
- Supported image formats: `.jpg`, `.png`, `.bmp`.

### Predict on Image:
- After training, you can use the model to predict steganography in new images.
- Click the "Predict on Image" button and select the image to be checked for hidden content.

### Exit:
- Once you're done using the tool, click on the "Exit" button to close the application.

## Image Directories:
- **Training Phase**: During the training phase, you will be prompted to select two directories:
  - One for clean images (images without steganography).
  - One for steganographic images (images that contain hidden data).
- Ensure the folders contain image files with `.jpg`, `.png`, or `.bmp` formats.

## Requirements
To run the application, ensure you have the following Python packages installed:

- Python 3.6+
- Tkinter
- OpenCV
- PIL (Pillow)
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TQDM

```bash
pip install -r requirements.txt
```
## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please feel free to fork the repository and create a pull request.

## How to contribute:
**Fork the repository**: Click the "Fork" button at the top of the page to create your own copy of the repository.

**Make changes**: Clone your forked repository and make your changes locally.

**Create a pull request**: Once you're done, push your changes to your forked repository and create a pull request to merge them into the original repository.

- **Kaggle Dataset**
This project uses the [StegoImagesDataset](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset) from Kaggle for training the model. Make sure to download the dataset before training the model.

## Acknowledgments:
### contributor:
Saffan Farooqui - k214781@nu.edu.pk
