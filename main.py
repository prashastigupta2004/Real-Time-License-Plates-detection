from flask import Flask, request, jsonify
from src.detect_plates import detect_plates

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the License Plate Detection API!"

@app.route('/detect', methods=['POST'])
def detect():
    image_path = request.json['image_path']
    detected_text = detect_plates(image_path)
    return jsonify({"Detected Text": detected_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
