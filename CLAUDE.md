# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kidney CT Scan AI Classifier project that uses deep learning to classify CT scan images into four categories: Cyst, Normal, Stone, and Tumor. The project consists of a trained MobileNetV2 model and a Streamlit web application for deployment.

## Architecture

- **Model Training** (`train_model.ipynb`): Downloads the CT Kidney Dataset from Kaggle and trains a MobileNetV2-based classifier
  - Uses MobileNetV2 with alpha=0.35 (lightweight version)
  - Input size: 96x96 RGB images
  - Output: 4 classes (Cyst, Normal, Stone, Tumor)
  - Model saved as HDF5 format in `model/kidney_model.h5`

- **Web Application** (`app.py`): Streamlit-based frontend for inference
  - Loads the trained model from `model/kidney_model.h5`
  - Accepts image uploads (JPG, JPEG, PNG)
  - Displays prediction with confidence scores and probability breakdown

## Commands

### Run the Streamlit app
```bash
streamlit run app.py
```

### Train the model (in Jupyter)
Run all cells in `train_model.ipynb`

## Dependencies

- TensorFlow/Keras
- Streamlit
- NumPy
- Pillow
- kagglehub (for dataset download during training)

## Key Configuration

- Image preprocessing: resize to 96x96, normalize to [0, 1]
- Model expects RGB images
- Class order: `["Cyst", "Normal", "Stone", "Tumor"]`
