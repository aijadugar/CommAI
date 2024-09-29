from textstat import flesch_reading_ease
from textblob import TextBlob
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# def evaluate_clarity(text):
#     score = flesch_reading_ease(text)
#     return min(score, 100)  # Cap the score at 100

def evaluate_clarity(text):
    score = flesch_reading_ease(text)
    return round(min(score, 100), 2)  


def evaluate_conciseness(text):
    sentences = len(re.findall(r'\.', text))
    words = len(text.split())
    if sentences == 0:
        return 0  
    avg_sentence_length = words / sentences
    return round(avg_sentence_length, 2)  


def evaluate_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)  
    subjectivity = round(blob.sentiment.subjectivity, 2)  
    return polarity, subjectivity


def evaluate_engagement(text):
    questions = len(re.findall(r'\?', text))
    sentences = len(re.findall(r'\.', text))
    if sentences == 0:
        return 0  # for avoiding division by zero
    engagement_score = questions / sentences
    return round(engagement_score, 2)  


def evaluate_grammar(text):
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
            'suggestion': match['replacements'],
            'message': match['message'],
            'offset': match['offset'],
            'length': match['length']
        }
        corrections.append(correction)

    # Calculate the grammar score
    total_words = len(text.split())
    error_count = len(corrections)
    
    if total_words > 0:
        grammar_score = max(0, 100 - (error_count / total_words * 100))
    else:
        grammar_score = 100  # Assume perfect score for empty text

    # Round the grammar score to 2 decimal places
    grammar_score = round(grammar_score, 2)

    return grammar_score, corrections


def evaluate_vocabulary_usage(text):
    words = text.split()
    if len(words) == 0:
        return 0  # Handle empty input
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    return round(diversity,2)

def evaluate_response_appropriateness(response, previous_text):
    if previous_text is None:
        return 0  # Handle case where there is no previous text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([response, previous_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def evaluate_politeness(text):
    polite_keywords = [
        "please", "thank you", "kindly", "would you mind", 
        "could you", "appreciate", "grateful", "sorry", "excuse me"
    ]
    text_lower = text.lower()
    polite_count = sum(keyword in text_lower for keyword in polite_keywords)
    total_keywords = len(polite_keywords)
    politeness_score = polite_count / total_keywords
    return round(politeness_score,2)

def evaluate_text(text):
    grammar_score, grammar_corrections = evaluate_grammar(text)
    results = {
        "Clarity": evaluate_clarity(text),
        "Conciseness": evaluate_conciseness(text),
        "Sentiment": evaluate_sentiment(text),
        "Engagement": evaluate_engagement(text),
        "Grammar and Spelling": grammar_score,
        "Grammar Corrections": grammar_corrections,
        "Vocabulary Usage": evaluate_vocabulary_usage(text),
        "Response Appropriateness": evaluate_response_appropriateness(text, previous_text=text),  # Call appropriately
        "Politeness": evaluate_politeness(text)
    }
    
    return results
