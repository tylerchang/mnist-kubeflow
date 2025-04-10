import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect
from PIL import Image
import io

# Set model load path (this will be mounted as a volume in Kubernetes)
MODEL_LOAD_PATH = "/models/mnist_model"

app = Flask(__name__)

# Load the model (will be loaded when the container starts)
print(f"Loading model from {MODEL_LOAD_PATH}")
model = tf.keras.models.load_model(MODEL_LOAD_PATH)
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html', image_data=None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    # Read and process the image
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    
    # Reshape for the model (add batch and channel dimensions)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class] * 100)
    
    return render_template(
        'index.html',
        prediction=int(predicted_class),
        confidence=confidence
    )

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)