from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import io

app = FastAPI(
    title="Wheat Disease Prediction API",
    description="API for predicting wheat leaf diseases from images",
    version="1.0.0"
)

# Load model
model = tf.keras.models.load_model("model/final_mobilenet.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.get("/")
def home():
    return {
        "message": "Wheat Disease Prediction API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Open image
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize to model input size
    image = image.resize((160, 160))

    # Convert to numpy array
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "filename": file.filename
    }