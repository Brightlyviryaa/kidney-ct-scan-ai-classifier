# Kidney CT Scan AI Classifier

Streamlit app backed by a MobileNetV2 model that classifies kidney CT scans into **Cyst, Normal, Stone,** and **Tumor** categories. A pretrained model is bundled (`model/kidney_model.h5`) so you can run the demo immediately.

## Features
- MobileNetV2 (alpha=0.35) trained on Kaggle's CT Kidney dataset with 96x96 RGB inputs.
- Streamlit UI with drag-and-drop upload, diagnosis label, confidence score, and probability breakdown.
- Ready-to-run model weights plus a reproducible training notebook (`train_model.ipynb`).
- Lightweight requirements for local inference; kagglehub support for dataset download during training.

## Project Structure
- `app.py` – Streamlit inference app.
- `model/kidney_model.h5` – Pretrained model weights.
- `train_model.ipynb` – Notebook to retrain the classifier (downloads `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone` via kagglehub).
- `Week13_Unguided.ipynb` – Additional analysis notebook from the project course work.
- `requirements.txt` – Runtime dependencies.

## Quickstart
1) Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2) Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3) Upload a CT scan image (JPG/PNG). The app resizes to 96x96, normalizes to [0,1], and returns the predicted class with probabilities for all four categories.

## Retraining (optional)
- Open `train_model.ipynb` in Jupyter/Colab.
- Ensure your Kaggle credentials are available so `kagglehub.dataset_download("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")` can fetch the dataset.
- The notebook trains the MobileNetV2-based classifier and saves the updated weights to `model/kidney_model.h5`, which the Streamlit app will load automatically.

## Notes
- Class order expected by the model and UI: `["Cyst", "Normal", "Stone", "Tumor"]`.
- Input must be RGB; grayscale images should be converted before inference.
