from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, jsonify, request
import requests as re
import time,io
from PIL import Image

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ResNet19Classifier:
    def __init__(self):
        self.weights=ResNet18_Weights.IMAGENET1K_V1
        self.model=resnet18(weights=self.weights)
        self.model.eval()
        self.preprocessor=self.weights.transforms()

    def predict(self,image):
        inp=self.preprocessor(image).unsqueeze(0)
        preds=self.model(inp).squeeze(0)
        result=[]
        for idx in list(preds.sort()[1])[-1:-6:-1]:
            result.append((self.weights.meta["categories"][idx]))

        return result


classifier=ResNet19Classifier()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "resnet18"}), 200

@app.route('/predict',methods=['POST'])
def predict_image():
    try:
        start_time = time.time()

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        logger.info(image_file)
        
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        image_data = image_file.read()
        image_data = Image.open(io.BytesIO(image_data))

        preds=classifier.predict(image_data)

        inference_time = time.time() - start_time

        result = {
            "predictions": preds,
            "inference_time": inference_time,
            "filename": image_file.filename
        }

        logger.info(f"Prediction completed in {inference_time:.3f}s for {image_file.filename}")

        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in URL prediction endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model": "ResNet18",
        "framework": "PyTorch",
        "input_size": "224x224",
        "classes": len(classifier.weights.meta["categories"]),
        "endpoints": ["/predict", "/health", "/info"]
    }), 200

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)