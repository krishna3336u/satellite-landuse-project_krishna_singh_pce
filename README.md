# Satellite Image Land-Use Classifier & Temporal Change Detector

## Project Overview
This project classifies satellite images into land-use categories and detects temporal change between two images using cosine similarity on deep embeddings.

## 🔗 Live Demo

👉 **[Click here to try the app]([https://YOUR-APP-URL.streamlit.app/](https://satellite-landuse-projectkrishnasinghpce-nqkqnqzbtlf4pd5dinhdd.streamlit.app/))**

## Features
- EuroSAT land-use classification
- Transfer learning with ResNet-18
- Two-phase fine-tuning
- UC Merced holdout evaluation
- Embedding-based change detection
- Streamlit dashboard

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run app:
   streamlit run app.py

## Required Files
Place the trained model in:
- checkpoints/resnet18_best_final.pt

Place metadata files in:
- metadata/class_to_idx.json
- metadata/idx_to_class.json
- metadata/change_detection_summary.json
