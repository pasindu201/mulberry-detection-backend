from flask import Flask, request, jsonify
from ultralytics import YOLO  
from PIL import Image
import io
import numpy as np
import cv2
import base64
from flask_cors import CORS
import os
import glob
import json
from pathlib import Path
from torchvision import models, transforms
import torch
import datetime
from torch import nn

app = Flask(__name__)
CORS(app) 

# Load your YOLOv10 model using the Ultralytics API 
model = YOLO('best.pt')

# Load ResNet-50 architecture
model_leaf = models.resnet50(pretrained=False)  # Don't use pretrained weights

# Modify the fully connected layer to match the number of classes (4 in your case)
model_leaf.fc = nn.Linear(model_leaf.fc.in_features, 4)

# Load the saved state dictionary
model_leaf.load_state_dict(torch.load('leaf_disease_resnet50.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model_leaf.eval()

# Define transformation (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def open_image_with_cv2(file):
    # Read the image from file as a numpy array
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Read the image in color (BGR)
    
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_rgb) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file keyword in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Open image using OpenCV and convert it to RGB
    img = open_image_with_cv2(file)
    
    # Run the model on the image using the Ultralytics API with save=True.
    # This will save the output images to a folder (e.g., runs/detect/exp, runs/detect/exp2, etc.)
    results = model(img, save=True)
    
    # Process predictions from the results (bounding boxes, confidence scores, etc.)
    pred_data = process_predictions(results)
    
    base_dir = "runs/detect"
    
    # List all output directories in base_dir
    output_dirs = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    if not output_dirs:
        return jsonify({"error": "No output directories found in base directory"}), 500

    latest_output_dir = output_dirs[-1]
    json_file_path = os.path.join(latest_output_dir, "output.json")

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(pred_data, json_file, indent=4)

    # Find JPEG files in the latest directory
    image_files = glob.glob(os.path.join(latest_output_dir, "*.jpg"))
    if not image_files:
        return jsonify({"error": "No saved image found in output directory"}), 500

    # Read the first saved image file and encode it in base64
    try:
        with open(image_files[0], "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return jsonify({'error': f'Error reading saved image: {str(e)}'}), 500

    # Return the prediction data and the generated image (as a base64-encoded string)
    return jsonify({
        "predictions": pred_data,
        "generated_image": encoded_image
    })

@app.route('/search-history', methods=['GET'])
def getSearchHistory():
    base_dir = "runs/detect"
    folder_dirs = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir)]
    history_records = []

    for folder_dir in folder_dirs:
        try:
            creation_time = os.stat(folder_dir).st_ctime
            creation_datetime = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")

            json_file_path = os.path.join(folder_dir, "output.json")
            if not os.path.exists(json_file_path):
                continue

            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            history_records.append({
                "timestamp": creation_datetime,
                "srilankan": data.get("srilankan", 0),
                "foreign": data.get("foreign", 0)
            })

        except Exception as e:
            print(f"Error processing {folder_dir}: {str(e)}")
            continue

    return jsonify(history_records)  # Move this outside the loop

@app.route('/leaf-disease', methods=['POST'])
def getLeafDiseasePrediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file keyword in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the image and preprocess it
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(img).unsqueeze(0) 

        # Perform inference
        with torch.no_grad():
            output = model_leaf(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Define labels for leaf disease classes
        leaf_disease_classes = ['Disease free', 'Leaf rust', 'Leaf spot', 'powdery mildew']

        predicted_label = leaf_disease_classes[predicted_class]
        return jsonify({
            'prediction': predicted_label
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

def process_predictions(results):
    try:
        boxes = results[0].boxes
        result_data = [box[-1] for box in boxes.data.cpu().numpy()]
        return {"srilankan": result_data.count(1.0), "foreign": result_data.count(0.0)}
    except Exception as e:
        return {"error": f"Error processing predictions: {str(e)}"}


if __name__ == '__main__':
    app.run(debug=True)
