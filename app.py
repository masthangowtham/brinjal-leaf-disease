from flask import Flask, render_template, request, redirect
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# TensorFlow Lite model path
MODEL_FILE_ID = '1frIKcMmUP4vbO989g3v7-_xYjn2KEFDc'
MODEL_PATH = 'models/brinjal_model.tflite'

# Download .tflite model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading TFLite model from Google Drive...")
    url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
    os.makedirs('models', exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download completed.")

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded and ready!")

# Class names
class_names = ['Aphids', 'bacterial_wilt', 'cercospora', 'collar_rot', 'colorado_bettle', 'little-leaf', 'mites', 'pest']
class_names_telugu = ['పెను బంక', 'వెల్లులి ఎరుపు కాళ్ళు', 'పొడిన మొక్క', 'ముళ్ళு చేద', 'కలరాడో బీటిల్ ఆకు పోటు', 'చిన్న ఆకు వ్యాధి', 'ఎర్ర నల్లి', 'చీడపురుగు']
class_names_tamil = ['ஆஃபிட்ஸ்','பெக்டீரியல் வில்ட்','செர்கோஸ்போரா','காலர் ராட்','கலராடோ பெட்டில்','லிட்டில்-லீஃப்','மைட்ஸ்','பெஸ்ட்']

# Default Gemini API Key (replace this with your actual default key)
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

    # Preprocess the image
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

    # Perform prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
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

# ✅ Render.com requires binding to PORT variable, not default localhost:5000
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render assigned port or 5000 as default
    app.run(host='0.0.0.0', port=port, debug=True)
