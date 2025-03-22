from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import csv
import os
import logging
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins (modify for production)

# Configuration
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"  # Replace with your SMTP server
SMTP_PORT = 587  # Replace with your SMTP port
EMAIL_ADDRESS = "evanjoseph573@gmail.com"
EMAIL_PASSWORD = "rmiwqdowmtfkcmnq"
RECIPIENT_EMAIL = "evanjoseph573@gmail.com"

def send_email_with_attachment(file_path, filename):
    """Send an email with the CSV file as an attachment."""
    try:
        # Create the email
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = f"Lima Bridge AHP Survey Results: {filename}"

        # Attach the CSV file
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={filename}",
            )
            msg.attach(part)

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully with attachment: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

# Utility Functions
def calculate_priority_weights(matrix):
    """Calculate priority weights from a pairwise comparison matrix."""
    try:
        matrix = np.array(matrix, dtype=float)
        column_sums = matrix.sum(axis=0)
        normalized_matrix = matrix / column_sums
        priority_weights = normalized_matrix.mean(axis=1)
        return priority_weights.tolist()
    except Exception as e:
        logger.error(f"Error calculating priority weights: {e}")
        raise

def calculate_consistency_ratio(matrix):
    """Calculate the consistency ratio for a pairwise comparison matrix."""
    try:
        matrix = np.array(matrix, dtype=float)
        priority_weights = calculate_priority_weights(matrix)
        consistency_vector = matrix.dot(priority_weights)
        lambda_max = np.sum(consistency_vector / priority_weights) / len(matrix)
        consistency_index = (lambda_max - len(matrix)) / (len(matrix) - 1)
        random_index = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49][len(matrix) - 1]
        return consistency_index / random_index if random_index != 0 else 0
    except Exception as e:
        logger.error(f"Error calculating consistency ratio: {e}")
        raise

# Routes
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

        # Validate responses
        if not responses or not all(isinstance(row, list) for row in responses):
            return jsonify({"error": "Invalid responses format"}), 400

        # Perform calculations
        priority_weights = calculate_priority_weights(responses)
        consistency_ratio = calculate_consistency_ratio(responses)

        # Generate CSV content
        csv_content = generate_csv(respondent_info, selected_bridge, responses, priority_weights, consistency_ratio)
        file_name = f"survey_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file_path = os.path.join(RESULTS_DIR, file_name)

        # Save CSV file
        with open(file_path, 'w') as file:
            file.write(csv_content)

        # Send CSV file as email attachment
        if send_email_with_attachment(file_path, file_name):
            return jsonify({
                "message": "Survey submitted successfully! CSV file sent via email.",
                "priorityWeights": priority_weights,
                "consistencyRatio": consistency_ratio
            }), 200
        else:
            return jsonify({"error": "Failed to send email"}), 500

    except Exception as e:
        logger.error(f"Error in submit_survey: {e}")
        return jsonify({"error": str(e)}), 500

def generate_csv(respondent_info, selected_bridge, responses, priority_weights, consistency_ratio):
    """Generate CSV content from survey data."""
    csv_content = f"Respondent Info: {respondent_info}\n"
    csv_content += f"Selected Bridge: {selected_bridge}\n\n"
    csv_content += "Responses Matrix:\n"
    for row in responses:
        csv_content += ",".join(map(str, row)) + "\n"
    csv_content += "\nPriority Weights:\n"
    for i, weight in enumerate(priority_weights):
        csv_content += f"Criterion {i + 1}: {weight}\n"
    csv_content += f"\nConsistency Ratio: {consistency_ratio}\n"
    return csv_content

@app.route('/save_csv_file', methods=['POST', 'OPTIONS'])
def save_csv_file():
    """Endpoint to save CSV file."""
    try:
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = jsonify({"status": "success"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "POST")
            return response, 200

        data = request.json
        if not data or 'csvContent' not in data:
            return jsonify({"error": "No CSV content provided"}), 400

        csv_content = data['csvContent']
        file_name = f"survey_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file_path = os.path.join(RESULTS_DIR, file_name)

        # Save CSV file
        with open(file_path, 'w') as file:
            file.write(csv_content)

        logger.info(f"CSV file saved successfully: {file_name}")
        return jsonify({"message": "CSV file saved successfully"}), 200

    except Exception as e:
        logger.error(f"Error in save_csv_file: {e}")
        return jsonify({"error": str(e)}), 500

# Main Entry Point
if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", "False") == "True", port=int(os.getenv("PORT", 10000)))
