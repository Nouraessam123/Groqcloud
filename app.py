from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# تحميل موديل من Hugging Face (مثلاً Flan-T5)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
analyzer = SentimentIntensityAnalyzer()

def generate_questions(role):
    prompt = f"Generate 3 behavioral and 2 technical interview questions for a {role} role. Just list the questions."
    result = qa_pipeline(prompt, max_new_tokens=200)[0]['generated_text']
    questions = [q.strip() for q in result.split("\n") if q.strip()]
    return questions

@app.route('/start-interview', methods=['POST'])
def start_interview():
    data = request.get_json()
    job_role = data.get('job_role', 'Data Scientist')
    questions = generate_questions(job_role)
    return jsonify({'questions': questions})

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    sentiment = analyzer.polarity_scores(answer)

    prompt = (
        f"Evaluate the following answer to this interview question:\n\n"
        f"Question: {question}\nAnswer: {answer}\n\n"
        f"Give detailed feedback and a score out of 10 like this:\nRating: X/10"
    )

    feedback = qa_pipeline(prompt, max_new_tokens=300)[0]['generated_text']

    rating = None
    lines = feedback.split('\n')
    for line in lines:
        if "Rating:" in line:
            import re
            match = re.search(r'\d+', line)
            if match:
                rating = int(match.group())
                break

    feedback_text = "\n".join([line for line in lines if not line.startswith("Rating:")])

    return jsonify({
        'feedback': feedback_text.strip(),
        'rating': rating,
        'sentiment': sentiment
    })
