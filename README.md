# Similar Image Retrieval using Autoencoder with VGG19
[By Erick Busuulwa at Nuist](https://github.com/busuulwa-ericklasco7)
This project demonstrates a Content-Based Image Retrieval (CBIR) system using an autoencoder architecture based on the VGG19 model. The system is capable of encoding images into a compact, meaningful representation for efficient image retrieval.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#discussion)
- [Future Work](#future-work)
- [License](#license)


## Introduction

Image retrieval is a crucial task in computer vision, where the goal is to find and retrieve images from a database that are similar or relevant to a given query image. This project leverages the VGG19 model and autoencoder architecture to create an effective image retrieval system.

## Features

- Uses VGG19 for feature extraction
- Employs an autoencoder for dimensionality reduction
- Efficiently retrieves similar images using the K-nearest neighbors (KNN) algorithm
- Visualizes the retrieval results and image embeddings


## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- skimage
- matplotlib
- Google Colab (if running on Colab)


## Usage

### Dataset Preparation

1. Collect and organize your dataset into training and testing directories as they are in the Dataset folder into you google drive.
2. Preprocess the images to ensure they have consistent dimensions and normalization.

### Code Execution

1. Clone this repository:
    ```sh
    git clone https://github.com/busuulwa-ericklasco7/Content-Based-Image-Retrieval-using-Autoencoder-with-VGG19.git
    cd VGGIM
    ```


2. Open the Colab notebook:
    [Similar_Image_Retrieval_using_autoencoder_using_vgg19.ipynb](https://colab.research.google.com/drive/1rzpDs4yyDxJbJw9iY_U_4WAgOK-Vg1dB)

3. Follow the instructions in the notebook to:
    - Mount your Google Drive (if using Colab)
    - Load and preprocess the dataset
    - Set up and train the autoencoder
    - Perform image retrieval and visualize results

### Important Functions

- `read_img(filePath)`: Reads an image from the given file path.
- `read_imgs_dir(dirPath, extensions, parallel=True)`: Reads all images from the given directory.
- `normalize_img(img)`: Normalizes the image values.
- `resize_img(img, shape_resized)`: Resizes the image to the given shape.
- `plot_query_retrieval(img_query, imgs_retrieval, outFile)`: Plots the query image and its retrieved similar images.

## Results

The results of the image retrieval system are demonstrated through visual examples of query images and their retrieved similar images. You can visualize these results by running the code in the provided Colab notebook.

## Discussion

### Analysis

The VGG19 model effectively extracts high-level features from images, which are essential for distinguishing between different images. The autoencoder helps reduce the dimensionality of these features while preserving their essential information for similarity assessment.

### Challenges

- Dataset quality and diversity significantly impact the retrieval performance for this task i used a simple dataset you might want to experiment with a large dataset.
- Hyperparameter tuning is crucial for achieving optimal performance.
- Computational resources are necessary for training deep learning models.

## Future Work

- Explore more advanced architectures, such as variational autoencoders (VAEs) or generative adversarial networks (GANs).
- Implement data augmentation techniques to improve model robustness.
- Optimize the model and retrieval process for real-time applications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.