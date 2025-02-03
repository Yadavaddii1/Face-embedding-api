from flask import Flask, request, jsonify
from keras_facenet import FaceNet
import numpy as np
import cv2

app = Flask(__name__)
embedder = FaceNet()

# Home Route to Check API Status
@app.route('/')
def home():
    return "Face Embedding API is Running!", 200


# 📌 Generate Face Embedding API
@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Ensure correct size
    image = cv2.resize(image, (160, 160))

    # Generate embedding
    embedding = embedder.embeddings([image])[0].tolist()

    return jsonify({'embedding': embedding})


# 📌 Cosine Similarity Function
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


# 📌 Compare Embeddings API
@app.route('/compare-embedding', methods=['POST'])
def compare_embedding():
    try:
        data = request.get_json()
        embedding1 = data.get('embedding1')
        embedding2 = data.get('embedding2')

        if not embedding1 or not embedding2:
            return jsonify({'error': 'Both embeddings are required'}), 400

        similarity = cosine_similarity(embedding1, embedding2)

        return jsonify({'similarity': similarity})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Run Flask App on Port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
