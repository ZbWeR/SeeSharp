from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        # 从POST请求中获取图片数据
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 使用DeepFace进行人脸表情识别
        result = DeepFace.analyze(img, actions=['emotion'])

        # 获取表情和置信度
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        response = {
            "emotion": emotion,
            "confidence": f"{confidence:.2f}%"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
