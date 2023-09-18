# server.py
from flask import Flask, request, jsonify
from app import process_question

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def get_gpt_response():
    transcript = request.form['transcript']
    text_response = process_question(transcript)
    return jsonify({'response': text_response})  # return the text response in a JSON format.

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)