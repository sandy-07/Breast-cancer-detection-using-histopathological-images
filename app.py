from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Limit TensorFlow memory usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

# Set the upload folder inside static
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "breast_cancer_model.keras"  # Changed to .keras format
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file '{MODEL_PATH}' not found.")

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "password":
            return "Login Successful!"
        else:
            return render_template('login.html', error="Invalid Credentials")

    return render_template('login.html')

# Fix for favicon.ico 404 error
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Predict Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Use the form to submit an image.", 405

    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
        except Exception as e:
            return f"File upload failed: {e}", 500

        try:
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize image

            if model:
                prediction = model.predict(img_array)
                print(f"Raw Prediction Value: {prediction[0][0]}")  # Debugging output

                threshold = 0.6  # Adjust based on testing
                result = "Benign" if prediction[0][0] < threshold else "Malignant"
            else:
                result = "Model not available"

            return render_template('home.html', prediction=result, image_url=url_for('static', filename=f'uploads/{filename}'))
        except Exception as e:
            return f"Error processing image: {e}", 500

    return "Invalid file format", 400

if __name__ == '__main__':
    app.run(debug=True)