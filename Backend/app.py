import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow debug logs
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"  # Helps reduce memory use
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"  # Limits TensorFlow memory to 50%
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimizations (reduces errors)

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Allow CORS for your frontend origin (Change to '*' if necessary)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the trained LSTM model
model = tf.keras.models.load_model("models.h5", compile=False)

# Compile the model (even if it’s already trained)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define the max sequence length used in training
MAX_SEQUENCE_LENGTH = 50  

@app.route('/')
def index():
    return "Welcome to Spam Detector API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        message = data.get("message", "").strip().lower()

        # Convert text into sequence & pad it
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        label = "Spam" if prediction > 0.5 else "Ham"

        # Return JSON response
        return jsonify({
            "message": message,
            "prediction": label,
            "confidence": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)