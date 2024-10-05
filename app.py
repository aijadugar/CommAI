from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from evaluation_parameters import evaluate_text
from level_selector import select_level
from grammer_spelling import evaluate_grammer_spelling

app = Flask(__name__)

# Chatbot setup
template = """
You are now conversational assistent for CommAI Project, make conversation with the user and answer the questions below in less words.
Here is the conversation history: {context}
Question: {question}
Answer:
"""
model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def text_interface():
    return render_template('text_interface.html')

@app.route('/speech')
def speech_interface():
    return render_template('speech_interface.html')

# Function to extract user messages from file
def extract_user_messages_from_file(file_path='conversations.json'):
    with open(file_path, 'r') as file:
        json_string = file.read().strip()

    json_string = json_string.replace('\\', '')
    user_messages = re.findall(r'"sender":"user","message":"(.*?)"', json_string)

    if not user_messages:
        return ''

    return '\n'.join(user_messages)

# Flask route for results page
@app.route('/results')
def results():
    # Extract user messages from the file
    user_text = extract_user_messages_from_file()

    # Evaluate the extracted user messages
    evaluation_results = evaluate_text(user_text)
    evaluation_level = select_level(user_text)
    grammer_spelling_evaluation = evaluate_grammer_spelling(user_text) 

    # Render the results page with evaluation data
    return render_template('results.html', gram_spell = grammer_spelling_evaluation, results=evaluation_results, level = evaluation_level)


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    conversation = request.json.get('conversation', '')


    # Append the user input to the conversation history
    conversation += f"\nUser: {user_input}"

    # Generate AI response using the model
    result = prompt | model
    ai_response = result.invoke({"context": conversation, "question": user_input})

    # Append the AI response to the conversation
    conversation += f"\nAI: {ai_response}"

    # Return the conversation and AI response as JSON
    return jsonify({'response': ai_response, 'conversation': conversation})

@app.route('/submit', methods=['POST'])
def submit():
    conversation = request.json.get('conversation')

    # Overwrite the previous conversation in the JSON file
    try:
        with open('conversations.json', 'w') as file:  # Use 'w' to overwrite
            json.dump(conversation, file)
        
        # Redirect to the results page with the conversation as a JSON string
        conversation_data = json.dumps(conversation)
        return jsonify({'status': 'success', 'redirect': f'/results?data={conversation_data}'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
