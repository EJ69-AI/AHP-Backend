from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)
CORS(app)

# Configuration
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "evanjoseph573@gmail.com"
EMAIL_PASSWORD = "rmiwqdowmtfkcmnq"
RECIPIENT_EMAIL = "evanjoseph573@gmail.com"

def send_email_with_attachment(file_path, filename):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = f"Lima Bridge AHP Survey Results: {filename}"

        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={filename}",
            )
            msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully with attachment: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def calculate_priority_weights(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        column_sums = matrix.sum(axis=0)
        normalized_matrix = matrix / column_sums
        priority_weights = normalized_matrix.mean(axis=1)
        return priority_weights / priority_weights.sum()  # Normalize to sum to 1
    except Exception as e:
        logger.error(f"Error calculating priority weights: {e}")
        raise

def calculate_consistency_ratio(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        priority_weights = calculate_priority_weights(matrix)
        consistency_vector = matrix.dot(priority_weights)
        lambda_max = np.sum(consistency_vector / priority_weights) / len(matrix)
        consistency_index = (lambda_max - len(matrix)) / (len(matrix) - 1)
        
        # Random Index values for different matrix sizes
        random_index_values = {
            2: 0,
            3: 0.58,
            4: 0.9,
            5: 1.12,
            6: 1.24,
            7: 1.32,
            8: 1.41,
            9: 1.45,
            10: 1.49
        }
        
        random_index = random_index_values.get(len(matrix), 1.49)
        return consistency_index / random_index if random_index != 0 else 0
    except Exception as e:
        logger.error(f"Error calculating consistency ratio: {e}")
        raise

@app.route('/submit', methods=['POST'])
def submit_survey():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        respondent_info = data.get('respondentInfo', {})
        responses = data.get('responses', {})

        # Validate responses
        required_levels = ['categories', 'technical', 'economic', 'environmental', 'social']
        if not all(level in responses for level in required_levels):
            return jsonify({"error": "Invalid responses format"}), 400

        # Calculate weights and CR for each level
        results = {
            "priorityWeights": {},
            "consistencyRatio": {}
        }

        for level in required_levels:
            matrix = responses[level]
            results["priorityWeights"][level] = calculate_priority_weights(matrix).tolist()
            results["consistencyRatio"][level] = float(calculate_consistency_ratio(matrix))

        # Generate CSV content
        csv_content = generate_csv(respondent_info, responses, results)
        file_name = f"survey_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file_path = os.path.join(RESULTS_DIR, file_name)

        with open(file_path, 'w') as file:
            file.write(csv_content)

        if send_email_with_attachment(file_path, file_name):
            return jsonify(results), 200
        else:
            return jsonify({"error": "Failed to send email"}), 500

    except Exception as e:
        logger.error(f"Error in submit_survey: {e}")
        return jsonify({"error": str(e)}), 500

def generate_csv(respondent_info, responses, results):
    csv_content = f"Respondent Info: {respondent_info.get('firstName', '')} {respondent_info.get('lastName', '')}\n"
    csv_content += f"Type: {respondent_info.get('type', '')}\n\n"
    
    # Category-level matrix
    csv_content += "Category-Level Comparisons\n"
    categories = ['Technical Indicators', 'Economic Indicators', 'Environmental Indicators', 'Social Indicators']
    csv_content += "," + ",".join(categories) + "\n"
    for i, row in enumerate(responses['categories']):
        csv_content += f"{categories[i]}," + ",".join(map(str, row)) + "\n"
    
    # Criteria-level matrices
    criteria_mapping = {
        'technical': 'Technical Indicators',
        'economic': 'Economic Indicators',
        'environmental': 'Environmental Indicators',
        'social': 'Social Indicators'
    }
    
    for level, category in criteria_mapping.items():
        csv_content += f"\n{category} Comparisons\n"
        criteria = get_criteria_for_level(level)
        csv_content += "," + ",".join(criteria) + "\n"
        for i, row in enumerate(responses[level]):
            csv_content += f"{criteria[i]}," + ",".join(map(str, row)) + "\n"
    
    # Priority weights
    csv_content += "\nPriority Weights\n"
    for level, weights in results['priorityWeights'].items():
        csv_content += f"\n{level.capitalize()} Weights\n"
        criteria = categories if level == 'categories' else get_criteria_for_level(level)
        for i, weight in enumerate(weights):
            csv_content += f"{criteria[i]},{weight:.4f}\n"
    
    # Consistency ratios
    csv_content += "\nConsistency Ratios\n"
    for level, cr in results['consistencyRatio'].items():
        csv_content += f"{level.capitalize()}: {cr:.4f}\n"
    
    return csv_content

def get_criteria_for_level(level):
    if level == 'technical':
        return ['Structural Feasibility', 'Constructability', 'Durability & Maintenance', 'Safety']
    elif level == 'economic':
        return ['Initial Cost', 'Lifecycle Cost', 'Return of investment', 'Construction Time']
    elif level == 'environmental':
        return ['Environmental Impact', 'Sustainability', 'Footprint']
    elif level == 'social':
        return ['Aesthetics & Cultural Value', 'Community Impact']
    return []

@app.route('/save_csv_file', methods=['POST'])
def save_csv_file():
    try:
        data = request.json
        if not data or 'csvContent' not in data:
            return jsonify({"error": "No CSV content provided"}), 400

        csv_content = data['csvContent']
        file_name = f"survey_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file_path = os.path.join(RESULTS_DIR, file_name)

        with open(file_path, 'w') as file:
            file.write(csv_content)

        logger.info(f"CSV file saved successfully: {file_name}")
        return jsonify({"message": "CSV file saved successfully"}), 200
    except Exception as e:
        logger.error(f"Error in save_csv_file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", "False") == "True", port=int(os.getenv("PORT", 10000)))
