from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from ClassificationModule import Classifier
from HandTrackingModule import HandDetector

app = Flask(__name__)
CORS(app)

# Cấu hình mô hình ASL
MODEL_PATH = "model_asl/keras_model.h5"
LABELS_PATH = "model_asl/labels.txt"
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)
offset = 20
imgSize = 300

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "ASL Recognition API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    hands = detector.findHands(img, draw=False)
    if not hands:
        return jsonify({"error": "No hand detected."}), 200

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

        predictions, index = classifier.getPrediction(imgWhite, draw=False)
        label = classifier.list_labels[index]
        confidence = float(predictions[index])

        # Tạo danh sách tất cả nhãn với độ chính xác
        all_predictions = [
            {"label": lbl, "confidence": round(float(conf), 4)}
            for lbl, conf in zip(classifier.list_labels, predictions)
        ]

        return jsonify({
            "top_prediction": {
                "label": label,
                "confidence": round(confidence, 4)
            },
            "all_predictions": all_predictions
        })

    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
