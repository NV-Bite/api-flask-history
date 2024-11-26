from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model/model_xception.keras", compile=False)
class_labels = ['Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
                'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
                'Rendang', 'Sate', 'Soto Ayam']

@app.route("/predict_image", methods=["POST"])
def predict_image():
    image = request.files.get("image")
    if image:
        img = Image.open(image).resize((300, 300))  # Resize gambar ke 300x300
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = xception_preprocess_input(img_array)

        # Lakukan prediksi
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = round(predictions[0][predicted_class_index] * 100, 1)  # Confidence score

        # Kembalikan respons JSON
        return jsonify({
            "data": {
                "predicted_class": predicted_class_label,
                "confidence": float(confidence),  # Konversi ke float
                "predict_time": 0.1  # Waktu prediksi (dummy)
            }
        }), 200

    return jsonify({"error": "Invalid image input"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
