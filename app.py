from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Google Drive File ID of the .h5 model
MODEL_FILE_ID = '1DFAISu8-zx9c9h-iD9wPngKQAoZhV1xx'
MODEL_PATH = 'models/brinjal_model.h5'

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
    os.makedirs('models', exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download completed.")

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Class names
class_names = ['Aphids', 'bacterial_wilt', 'cercospora', 'collar_rot', 'colorado_bettle', 'little-leaf', 'mites', 'pest']
class_names_telugu = ['పెను బంక', 'వెల్లులి ఎరుపు కాళ్ళు', 'పొడిన మొక్క', 'ముళ్ళு చేద', 'కలరాడో బీటిల్ ఆకు పోటు', 'చిన్న ఆకు వ్యாதி', 'ఎర్ర నల్లి', 'చీడపురుగు']
class_names_tamil = ['ஆஃபிட்ஸ்','பெக்டீரியல் வில்ட்','செர்கோஸ்போரா','காலர் ராட்','கலராடோ பெட்டில்','லிட்டில்-லீஃப்','மைட்ஸ்','பெஸ்ட்']

DEFAULT_API_KEY = "YOUR_DEFAULT_GEMINI_API_KEY"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    language = request.form.get('language')
    user_api_key = request.form.get('api_key') or DEFAULT_API_KEY

    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_label = class_names[predicted_class_idx]

    # Translated label
    label_tamil = class_names_tamil[predicted_class_idx]
    label_telugu = class_names_telugu[predicted_class_idx]
    translated_name = label_tamil if language == 'tamil' else label_telugu if language == 'telugu' else predicted_class_label

    return render_template('result.html',
                           disease_name=predicted_class_label,
                           translated_name=translated_name,
                           selected_language=language.upper(),
                           api_key_used=user_api_key,
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
