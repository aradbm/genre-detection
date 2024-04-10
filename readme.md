# Music Genre Detection Using MP3 & WAV Files

## Overview

This project explores music genre classification using different machine learning algorithms. Starting from 2 datasets containing audio files in MP3 and WAV formats, we use feature extraction techniques to prepare the data for the application of several classification algorithms, with the goal of accurately identifying the music genre of each song. The algorithms used include:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Neural Network

## Documents

- [Project Description](/docs/Project_Description.png)
- [Project Proposal](/docs/Project_Proposal.pdf)
- [Project Presentation](/docs/Project_Presentation.pdf)
- [Project Log](/docs/Project_Log.txt)

## Setup

To run this project, ensure Python 3.x is installed on your machine. Additionally, the following Python libraries are required:

- librosa
- numpy
- pandas
- scikit-learn

These can be installed via pip using the command:
pip install librosa numpy pandas scikit-learn

## Results

The classification algorithms' performance is detailed in the `results/` directory. This includes a comparison of the algorithms based on 20 runs, evaluating them on their ability to classify music based on chroma features and Mel-Frequency Cepstral Coefficients (MFCCs). Features csv result files are included in `features/chroma` or `features/mfcc` directories.

## Datasets

The project utilizes two primary datasets, described below. Both datasets are preprocessed and categorized by music genres to streamline the training and evaluation process.

1. **GTZAN Dataset - Music Genre Classification**: A benchmark dataset for music genre classification comprising 1000 audio tracks, 30 seconds each, across 10 genres. Available on Kaggle and stored in `data/gen-dataset`.

   - Kaggle Link: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

2. **Emotify Dataset - Emotion Classification in Songs**: Provides a unique perspective on genre classification through emotion recognition, containing a diverse array of musical genres. Available on Kaggle and stored in `data/emo-dataset`.
   - Kaggle Link: [Emotify Dataset](https://www.kaggle.com/datasets/yash9439/emotify-emotion-classificaiton-in-songs)

### Data Preprocessing

Feature extraction processes involve converting MP3 files to WAV format, followed by the extraction of Mel-Frequency Cepstral Coefficients (MFCCs) and Chroma features. The scripts for these procedures are located in `src/feature_extraction` directory.

## Running the Project

Follow these steps to run the project:

1. **Download Datasets**: Acquire and preprocess the data as described above.
2. **Feature Extraction**: Execute the feature extraction scripts within the `src/feature_extraction` directory.
3. **Training and Evaluation**: Proceed to `src/ml_algorithms` and run the script for the desired machine learning algorithm. For instance, to utilize the SVM classifier: `python svm.py`.

Optional: For more details on the project's progress, refer to the documentation in the docs directory, including project logs, descriptions, and proposals.

## Contributors

- Eliyahu Greenblatt
- Arad Ben Menashe
