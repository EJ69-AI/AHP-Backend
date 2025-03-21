from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import csv
import os
import sys

# Define Flask app
app = Flask(__name__)
CORS(app, origins=["https://ahp-frontend.vercel.app"])

# Print registered routes AFTER defining app
print("Registered Routes:", app.url_map, file=sys.stderr)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return "API is running!"

@app.route('/submit', methods=['POST'])
def submit_survey():
    try:
        data = request.json
        respondent_name = data.get("name", "Anonymous")
        pairwise_comparisons = data.get("comparisons", [])

        file_path = os.path.join(RESULTS_DIR, "survey_results.csv")
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([respondent_name] + pairwise_comparisons)

        return jsonify({"message": "Survey submitted successfully!"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calculate_priority_weights(matrix):
    matrix = np.array(matrix, dtype=float)
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    priority_weights = normalized_matrix.mean(axis=1)
    return priority_weights.tolist()

def calculate_consistency_ratio(matrix):
    matrix = np.array(matrix, dtype=float)
    priority_weights = calculate_priority_weights(matrix)
    consistency_vector = matrix.dot(priority_weights)
    lambda_max = np.sum(consistency_vector / priority_weights) / len(matrix)
    consistency_index = (lambda_max - len(matrix)) / (len(matrix) - 1)
    random_index = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49][len(matrix) - 1]
    return consistency_index / random_index if random_index != 0 else 0

@app.route('/save_csv', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        respondent_info = data.get('respondentInfo', {})
        first_name = respondent_info.get('firstName', 'Unknown')
        last_name = respondent_info.get('lastName', 'Unknown')
        bridge_option = data.get('selectedBridge', 'N/A')
        matrix = data.get('responses', [])

        if not matrix or not all(isinstance(row, list) for row in matrix):
            return jsonify({"error": "Matrix data missing or invalid"}), 400

        priority_weights = calculate_priority_weights(matrix)
        consistency_ratio = calculate_consistency_ratio(matrix)

        filename = f"{RESULTS_DIR}/{first_name}_{last_name}.csv"
        df = pd.DataFrame(matrix)
        df.to_csv(filename, index=False)

        return jsonify({
            "message": "CSV saved successfully",
            "bridgeOption": bridge_option,
            "priorityWeights": priority_weights,
            "consistencyRatio": consistency_ratio
        }), 200

    except Exception 
