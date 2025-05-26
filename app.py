from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from ClassificationModule import Classifier
from UTF8ClassificationModule import UTF8Classifier
from HandTrackingModule import HandDetector

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Khởi tạo mô hình nhận diện ASL (hoặc thay thế bằng model bạn muốn)
MODEL_PATH = "model_asl/keras_model.h5"
LABELS_PATH = "model_asl/labels.txt"

# Dò tìm tay và phân loại ký hiệu
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)
offset = 20
imgSize = 300

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Sign Language Recognition API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    hands = detector.findHands(img, draw=False)
    if not hands:
        return jsonify({"error": "No hand detected"}), 200

    hand = hands[0]
    x, y, w, h = hand["bbox"]
    try:
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = classifier.list_labels[index]
        confidence = float(prediction[index])

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
