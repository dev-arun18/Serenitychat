from flask import Flask, request, jsonify, render_template
from data_preparation import get_response

# Initialize Flask app
app = Flask(__name__)

# Flask route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle the chat functionality
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    return jsonify({'response': response})

# Main block to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
