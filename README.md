# BMI-Decoder-Comparison
This repository contains the Python code used in the experiments of our research paper:  
[**Comparative Analysis of Neural Decoding Algorithms for Brain-Machine Interfaces**](https://www.biorxiv.org/content/10.1101/2024.12.05.627080v1.abstract)
**Authors:** Olena Shevchenko, Sofiia Yeremeieva, and Brokoslaw Laschowski  

## **Overview**
Brain-machine interfaces (BMIs) rely on accurate neural decoding algorithms to interpret brain activity. This project systematically evaluates different combinations of state-of-the-art **signal processing, feature extraction, and classification algorithms** to determine the optimal combination for motor neural decoding using EEG data.

The repository is structured into different modules to facilitate **signal processing, feature extraction, and classification**.

## **1. Dataset**
We used the publicly available EEG dataset:  
[**A Walk in the Park: Characterizing Gait-Related Artifacts in Mobile EEG Recordings**](https://www.researchgate.net/publication/344222835_A_walk_in_the_park_Characterizing_gait-related_artifacts_in_mobile_EEG_recordings)

## **2. Repository Structure**
BMI-Decoder-Comparison/
│── BMIResearch/                # General research-related scripts\
│── constants/                   # Constants used throughout the project\
│   ├── constants.py\
│── notebooks/                   # Jupyter Notebooks for each step of the pipeline\
│   ├── Signal_Processing/       # Preprocessing EEG signals\
│   │   ├── ASR.ipynb            # Artifact Subspace Reconstruction\
│   │   ├── SLF.ipynb            # Surface Laplacian Filter\
│   ├── Feature_Extraction/      # Extracting relevant features from EEG\
│   │   ├── CSP.ipynb            # Common Spatial Patterns\
│   │   ├── ICA.ipynb            # Independent Component Analysis\
│   │   ├── STFT.ipynb           # Short-Time Fourier Transform\
│   ├── Classification/          # Machine learning models for classification\
│   │   ├── SVM.ipynb            # Support Vector Machine\
│   │   ├── LDA.ipynb            # Linear Discriminant Analysis\
│   │   ├── EEGNet_cnn.ipynb     # Convolutional Neural Network (EEGNet)\
│   │   ├── LSTM.ipynb           # Long Short-Term Memory (LSTM)\
│   ├── create_dataset.ipynb     # Dataset creation and preparation\
│   ├── Results.ipynb            # Evaluation and results analysis\
│── pipeline_structure/          # Scripts for modular pipeline processing\
│   ├── signal_processing/       # Preprocessing utilities\
│   │   ├── asr_utils.py         # Utility functions for ASR processing

## **3. Signal Processing Methods**
We applied state-of-the-art signal processing techniques to improve EEG data quality:
- **Artifact Subspace Reconstruction (ASR)** – Automatically removes transient artifacts from EEG signals.
- **Surface Laplacian Filter (SLF)** – Enhances spatial resolution and suppresses volume conduction effects.

## **4. Feature Extraction Methods**
To extract meaningful neural information, we compared:
- **Common Spatial Patterns (CSP)** – Optimized spatial filters for distinguishing brain activity.
- **Independent Component Analysis (ICA)** – Isolates independent signal sources from EEG.
- **Short-Time Fourier Transform (STFT)** – Converts EEG signals into time-frequency representations.

## **5. Classification Models**
We evaluated both **deep learning** and **classical machine learning** classifiers:
- **Support Vector Machine (SVM)**
- **Linear Discriminant Analysis (LDA)**
- **Convolutional Neural Network (CNN - EEGNet)**
- **Long Short-Term Memory (LSTM)**
