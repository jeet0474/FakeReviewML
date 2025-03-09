from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
from urllib.parse import unquote
from flask_cors import CORS  # Import CORS to allow frontend requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DJANGO_API_URL = "https://fakereviewml-backend.onrender.com/api/predict"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'verified.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_reviews', methods=['GET'])
def fetch_reviews():
    flipkart_url = request.args.get('url', '')

    if not flipkart_url:
        return jsonify({'error': 'No URL provided'}), 400

    flipkart_url = unquote(flipkart_url)  # Decode the URL
    full_api_url = f"{DJANGO_API_URL}?url={flipkart_url}"

    try:
        response = requests.get(full_api_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data.get('text'):
            return jsonify({'error': 'No reviews found in response'}), 404

        return jsonify(data)

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error fetching from Django: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
