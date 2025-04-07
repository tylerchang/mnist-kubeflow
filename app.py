import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from PIL import Image
import io
import base64

# Set model load path (this will be mounted as a volume in Kubernetes)
MODEL_LOAD_PATH = "/models/mnist_model"

app = Flask(__name__)

# Load the model (will be loaded when the container starts)
print(f"Loading model from {MODEL_LOAD_PATH}")
model = tf.keras.models.load_model(MODEL_LOAD_PATH)
print("Model loaded successfully!")

# Simple HTML template for the UI with image upload
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Digit Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            text-align: center;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        .preview {
            margin: 20px 0;
            border: 1px solid #ddd;
            padding: 10px;
        }
        #result {
            font-size: 24px;
            margin-top: 20px;
        }
        button, input[type="submit"] {
            padding: 10px 20px;
            margin: 10px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover, input[type="submit"]:hover {
            background-color: #45a049;
        }
        .image-preview {
            max-width: 280px;
            max-height: 280px;
            margin: 10px auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MNIST Digit Recognition</h1>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div>
                <input type="file" name="image" id="imageInput" accept="image/*">
                <input type="submit" value="Upload & Predict">
            </div>
        </form>
        
        {% if image_data %}
        <div class="preview">
            <h3>Uploaded Image</h3>
            <img src="data:image/png;base64,{{ image_data }}" class="image-preview">
            <div id="result">Prediction: {{ prediction }} (Confidence: {{ confidence|round(2) }}%)</div>
        </div>
        {% else %}
        <div class="preview">
            <h3>Upload an image of a handwritten digit (0-9)</h3>
        </div>
        {% endif %}
    </div>

    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                // Show form submit button when file is selected
                document.querySelector('input[type="submit"]').style.display = 'inline-block';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, image_data=None)

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
    
    # Check if we need to invert the image (MNIST has white digits on black)
    # If the average pixel value is high (bright image), invert it
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array
    
    # Reshape for the model (add batch and channel dimensions)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class] * 100)
    
    # Convert the processed image to base64 for display
    processed_img = (img_array[0, :, :, 0] * 255).astype(np.uint8)
    processed_pil = Image.fromarray(processed_img)
    buffered = io.BytesIO()
    processed_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return render_template_string(
        HTML_TEMPLATE, 
        image_data=img_str,
        prediction=int(predicted_class),
        confidence=confidence
    )

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)