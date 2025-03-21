from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import csv
import os
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins (modify for production)

# Configuration
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def calculate_priority_weights(matrix):
    """Calculate priority weights from a pairwise comparison matrix."""
    matrix = np.array(matrix, dtype=float)
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    priority_weights = normalized_matrix.mean(axis=1)
    return priority_weights.tolist()

def calculate_consistency_ratio(matrix):
    """Calculate the consistency ratio for a pairwise comparison matrix."""
    matrix = np.array(matrix, dtype=float)
    priority_weights = calculate_priority_weights(matrix)
    consistency_vector = matrix.dot(priority_weights)
    lambda_max = np.sum(consistency_vector / priority_weights) / len(matrix)
    consistency_index = (lambda_max - len(matrix)) / (len(matrix) - 1)
    random_index = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49][len(matrix) - 1]
    return consistency_index / random_index if random_index != 0 else 0

# Routes
@app.route('/')
def home():
    """Root endpoint."""
    return jsonify({"message": "Welcome to the AHP Backend!"}), 200

@app.route('/submit', methods=['POST'])
def submit_survey():
    """Endpoint to submit survey data."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        respondent_info = data.get('respondentInfo', {})
        selected_bridge = data.get('selectedBridge', 'N/A')
        responses = data.get('responses', [])

        # Perform calculations
        priority_weights = calculate_priority_weights(responses)
        consistency_ratio = calculate_consistency_ratio(responses)

        return jsonify({
            "message": "Survey submitted successfully!",
            "priorityWeights": priority_weights,
            "consistencyRatio": consistency_ratio
        }), 200

    except Exception as e:
        logger.error(f"Error in submit_survey: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_csv_file', methods=['POST'])
def save_csv_file():
    """Endpoint to save raw CSV content to a file."""
    try:
        data = request.get_json()
        if not data or 'csvContent' not in data:
            return jsonify({"error": "Invalid JSON or missing CSV content"}), 400

        csv_content = data['csvContent']
        filename = f"{RESULTS_DIR}/survey_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

        with open(filename, 'w') as file:
            file.write(csv_content)

        return jsonify({"message": "CSV file saved successfully"}), 200

    except Exception as e:
        logger.error(f"Error in save_csv_file: {e}")
        return jsonify({"error": str(e)}), 500

# Main Entry Point
if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", "False") == "True", port=int(os.getenv("PORT", 10000)))
