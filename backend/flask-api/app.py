from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image  # For handling images

app = Flask(__name__)

# Load the model outside the prediction function for efficiency
model_path = 'model/skin_cancer_model.h5'  # Adjust path if needed
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None  # Handle the case where the model isn't loaded

def load_and_preprocess_image(image_path, target_size):
    """Loads, resizes, and preprocesses a single image."""
    try:
        img = Image.open(image_path).convert("RGB")  # Use PIL for better handling
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500  # Server error

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400  # Bad request

    img_file = request.files['file']
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_path = "temp_image.jpg"  # Temporary filename
        img_file.save(image_path)

        target_size = (128, 128)  # Match your model's input size
        processed_image = load_and_preprocess_image(image_path, target_size)
        os.remove(image_path)  # Remove the temporary file

        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        prediction = model.predict(processed_image)

        if prediction[0][0] > 0.5:
            result = 'Malignant'
        else:
            result = 'Benign'

        return jsonify({'prediction': result, 'confidence': float(prediction[0][0])}), 200  # OK

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Server error

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 80)))