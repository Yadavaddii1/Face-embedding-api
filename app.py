from flask import Flask, request, jsonify
from keras_facenet import FaceNet
import numpy as np
import cv2
import os  # Import os for environment variable handling

app = Flask(__name__)
embedder = FaceNet()

# Home Route to Check API Status
@app.route('/')
def home():
    return "Face Embedding API is Running!", 200

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

# 📌 Generate Face Embedding API
@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Ensure correct size
        image = cv2.resize(image, (160, 160))

        # Generate embedding
        embedding = embedder.embeddings([image])[0].tolist()

        return jsonify({'embedding': embedding})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

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

# ✅ Run Flask App on Dynamic Port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Get the port from environment variables or use 5000 by default
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


