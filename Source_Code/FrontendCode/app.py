from flask import Flask, request, render_template, jsonify
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io


app = Flask(__name__)

# Load the model
model = load_model('cnn_densenet201_model.h5')
print("Expected input shape:", model.input_shape)  # Debugging input shape

# Define a dictionary to map indices to disease names
disease_classes = {
    0: 'Apple_Aphis_spp',
    1: 'Apple_Eriosoma_lanigerum',
    2: 'Apple_Monillia_laxa',
    3: 'Apple Venturia inaequalis',
    4: 'Apricot Coryneum beijerinckii',
    5: 'Apricot Monillia laxa',
    6: 'Cancer symptom',
    7: 'Cherry Aphis spp',
    8: 'Drying symptom',
    9: 'Peach Monillia laxa',
    10: 'Peach Parthenolecanium corni',
    11: 'Pear Erwinia amylovora',
    12: 'Plum Aphis spp',
    13: 'Walnut Eriophyes erineus',
    14: 'Walnut Gnomonia leptostyla'
}

# Preprocess the image according to model input requirements


def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match AlexNet input size
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=1)  # Add extra dimension
    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            disease_name = disease_classes.get(predicted_class, "Unknown")

            return jsonify({
                "prediction_index": int(predicted_class),
                "disease_name": disease_name
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file"}), 400


if __name__ == '__main__':
    app.run(debug=True)
