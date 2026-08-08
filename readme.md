# Wheat Disease Prediction API

A simple image prediction API built using **FastAPI** and **TensorFlow**. The API accepts a wheat leaf image and uses a trained MobileNetV2 model to predict the corresponding wheat disease.

## Features

- Upload a wheat leaf image through an API
- Image preprocessing and resizing
- Deep learning-based disease classification
- Returns predicted disease and confidence score
- Interactive API documentation using Swagger UI

## Tech Stack

- Python
- FastAPI
- TensorFlow / Keras
- MobileNetV2
- Pillow
- NumPy
- Uvicorn

## Project Structure

```text
WHEAT-DISEASE-PREDICTION/
│
├── model/
│   └── final_mobilenet.keras
│
├── app.py
├── class_names.json
├── requirements.txt
└── README.md
