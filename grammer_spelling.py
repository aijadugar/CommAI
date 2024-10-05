import requests
def evaluate_conversation_grammar(text):

    url = "https://api.languagetool.org/v2/check"

    data = {
        'text': text,
        'language': 'en-US',
    }

    response = requests.post(url, data=data)
    result = response.json()

    corrections = []
    for match in result.get('matches', []):
        correction = {
            'error': match['context']['text'],
            'suggestion': match['replacements'][:6],
            'message': match['message'],
            'offset': match['offset'],
            'length': match['length']
        }
        corrections.append(correction)

    return corrections

def evaluate_grammer_spelling(text):
    if text is None:
        return {
            "Grammar and Spelling": 0
        }
    
    gram_spell = {
        "Grammar and Spelling": evaluate_conversation_grammar(text)
    }
    
    return gram_spell



