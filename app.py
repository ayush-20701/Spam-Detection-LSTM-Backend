from flask import Flask, request, jsonify
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Load the trained LSTM model
model = tf.keras.models.load_model("models.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define the max sequence length used in training
MAX_SEQUENCE_LENGTH = 50  

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
    app.run()